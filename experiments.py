from __future__ import annotations

import math
import time
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.inspection import permutation_importance

# scipy ships with scikit-learn, so these are always available.
from scipy.stats import spearmanr, kendalltau


# ---------------------------------------------------------------------------
# 0.  Global configuration, reproducibility and plotting style
# ---------------------------------------------------------------------------

COLOR_SCHEME = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple/pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

METHOD_COLORS = {
    "usefulness":  COLOR_SCHEME[0],
    "shap":        COLOR_SCHEME[1],
    "permutation": COLOR_SCHEME[2],
    "lime":        COLOR_SCHEME[3],
}

PRETTY_NAMES = {
    "california":   "California Housing",
    "bike_sharing": "Bike Sharing",
    "adult_income": "Adult Income",
}


@dataclass
class Config:
    """All knobs of the experiments in one reproducible place."""

    seed: int = 42                       # master seed
    n_models: int = 20                   # trees trained per configuration (paper uses 20)
    bins_grid: Tuple[int, ...] = (3, 4, 5, 6)
    strategies: Tuple[str, ...] = ("uniform", "quantile", "kmeans")
    test_size: float = 0.20
    # per-dataset leaf regularization, matching the paper (100*bins / 150*bins).
    leaves_per_bin: Dict[str, int] = field(
        default_factory=lambda: {"california": 100, "bike_sharing": 100, "adult_income": 150}
    )
    # How many instances to use for the (sampling-based) importance methods.
    importance_sample: int = 200
    # Persistence parameter of Rank-Biased Overlap (higher -> weighs the whole
    # ranking more evenly; lower -> weighs only the very top of the ranking).
    rbo_p: float = 0.9
    results_dir: str = "results_ext"


def set_all_seeds(seed: int) -> None:
    """Seed every RNG we rely on, for run-to-run reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def set_plot_style() -> None:
    """A clean, paper-friendly matplotlib style (color-blind safe, legible).

    Typography matches a LaTeX-typeset paper: every piece of text (titles,
    labels, ticks, legends) is rendered *through* a real LaTeX install
    (Computer Modern), via matplotlib's ``text.usetex``.  Falls back to
    matplotlib's bundled "cm" mathtext fonts (still LaTeX-like, but with no
    external dependency) if no ``latex`` binary is found on PATH, so the
    module still works on a machine without a TeX distribution.
    """
    import shutil
    has_latex = shutil.which("latex") is not None
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "text.usetex": has_latex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=COLOR_SCHEME),
    })
    if not has_latex:
        warnings.warn("No system LaTeX install found ('latex' not on PATH); falling back "
                      "to matplotlib's bundled Computer-Modern-like fonts.")


# ---------------------------------------------------------------------------
# 1.  Internal decision-tree representation and scoring
# ---------------------------------------------------------------------------

class Node:
    """A node of a binary decision tree.

    Internal nodes carry a ``feature`` and a ``threshold`` (test ``feature <=
    threshold``); leaves carry the sentinel feature ``"True"``/``"False"``.
    """

    __slots__ = ("feature", "threshold", "left", "right")

    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    # --- accessors kept for parity with the notebook API --------------------
    def get_feature(self):            return self.feature
    def set_feature(self, new):       old, self.feature = self.feature, new; return old
    def get_threshold(self):          return self.threshold
    def set_threshold(self, new):     old, self.threshold = self.threshold, new; return old
    def get_left(self):               return self.left
    def set_left(self, node):         self.left = node
    def get_right(self):              return self.right
    def set_right(self, node):        self.right = node

    def is_leaf(self):                return self.left is None
    def is_true(self):                return self.feature == "True"
    def is_false(self):               return not self.is_true()


def _clone_node(n: "Node") -> "Node":
    """Fast recursive node copy (leaves share nothing; structure is duplicated)."""
    if n.left is None:
        return Node(n.feature, n.threshold, None, None)
    return Node(n.feature, n.threshold, _clone_node(n.left), _clone_node(n.right))


class Tree:
    """A decision tree over categorical features, with a model-counting oracle.

    ``domains[f] = (lo, hi)`` gives the inclusive integer range of feature ``f``.
    ``count()`` returns the number of integer entities classified as ``True``.
    """

    def __init__(self, root: Node, domains: Dict[str, Tuple[float, float]]):
        self.root = root
        self.domains = domains

    def get_root(self):
        return self.root

    # -- model counting ------------------------------------------------------
    def _count_recursive(self, node: Node, ranges: Dict[str, Tuple[float, float]]) -> int:
        if node.is_leaf() and node.is_true():
            # Multiply the number of integer values still available per feature.
            acum = 1
            for (a, b) in ranges.values():
                if b < a:
                    return 0
                acum *= (math.floor(b) - math.ceil(a) + 1)
            return acum

        elif not node.is_leaf():
            acum = 0
            cur_feature = node.get_feature()
            cur_threshold = node.get_threshold()
            a, b = ranges[cur_feature]

            # left branch: feature <= threshold
            ranges[cur_feature] = (a, min(cur_threshold, b))
            acum += 0 if min(cur_threshold, b) < a else self._count_recursive(node.get_left(), ranges)

            # right branch: feature > threshold  (tiny epsilon to move off the split)
            ranges[cur_feature] = (max(cur_threshold + 1e-7, a), b)
            acum += 0 if b < max(cur_threshold + 1e-7, a) else self._count_recursive(node.get_right(), ranges)

            ranges[cur_feature] = (a, b)   # restore for the caller
            return acum
        else:
            return 0

    def count(self) -> int:
        ranges = {f: self.domains[f] for f in self.domains}
        return self._count_recursive(self.root, ranges)

    # -- boolean algebra on trees -------------------------------------------
    def _negate_recursive(self, node: Node) -> None:
        if node.is_leaf():
            node.set_feature("False" if node.get_feature() == "True" else "True")
        else:
            self._negate_recursive(node.get_left())
            self._negate_recursive(node.get_right())

    def negate(self) -> None:
        self._negate_recursive(self.root)

    def _conjunction_recursive(self, node, prev_node, other_tree) -> None:
        if node.is_leaf():
            if node.is_true():
                if prev_node.get_right() is node:
                    prev_node.set_right(other_tree.get_root())
                else:
                    prev_node.set_left(other_tree.get_root())
        else:
            self._conjunction_recursive(node.get_left(), node, other_tree)
            self._conjunction_recursive(node.get_right(), node, other_tree)

    def conjunction(self, other_tree: "Tree") -> None:
        """Replace every ``True`` leaf with ``other_tree`` (destructive AND).

        Note: references, not copies, are grafted in, so a given ``other_tree``
          object must not be conjoined into two places.
        """
        self._conjunction_recursive(self.root, None, other_tree)

    def _condition_recursive(self, node: Node, feature: str, value: float) -> None:
        if node.is_leaf():
            return
        if node.get_feature() == feature:
            # Force the branch that ``feature == value`` would take.
            if value <= node.get_threshold():
                node.set_threshold(1e9)    # test always true  -> always go left
            else:
                node.set_threshold(-1)     # test always false -> always go right
        self._condition_recursive(node.get_left(), feature, value)
        self._condition_recursive(node.get_right(), feature, value)

    def condition(self, feature: str, value: float) -> None:
        """Restrict the tree to entities with ``feature == value``."""
        self._condition_recursive(self.root, feature, value)

    def clone(self) -> "Tree":
        """A working copy for scoring: the *nodes* are duplicated (they get
        mutated by ``condition``/``negate``/``conjunction``) but the immutable
        ``domains`` dict is **shared**.  This is ~5x faster than ``copy.deepcopy``
        and yields identical model counts (see ``profile_scorer_speedup``)."""
        return Tree(_clone_node(self.root), self.domains)

    def print(self, node: Optional[Node] = None, tabs: int = 0) -> None:
        node = self.root if node is None else node
        if node is None:
            return
        print(" " * tabs + f"{node.get_feature()} (<= {node.get_threshold()})")
        if not node.is_leaf():
            self.print(node.get_left(), tabs + 3)
            self.print(node.get_right(), tabs + 3)


def _dec_tree_to_my_tree_rec(node: int, dec_tree, features: Sequence[str]) -> Node:
    if dec_tree.tree_.feature[node] != -2:                       # internal node
        feature_name = features[dec_tree.tree_.feature[node]]
        left = _dec_tree_to_my_tree_rec(dec_tree.tree_.children_left[node], dec_tree, features)
        right = _dec_tree_to_my_tree_rec(dec_tree.tree_.children_right[node], dec_tree, features)
        threshold = dec_tree.tree_.threshold[node]
        return Node(feature_name, threshold, left, right)
    # leaf: majority class -> True/False sentinel leaf
    value = dec_tree.tree_.value[node]
    predicted_class = int(np.argmax(value))
    return Node("True", 1, None, None) if predicted_class == 1 else Node("False", 0, None, None)


def dec_tree_to_my_tree(dec_tree: DecisionTreeClassifier,
                        domains: Dict[str, Tuple[float, float]],
                        features: Sequence[str]) -> Tree:
    """Translate an sklearn ``DecisionTreeClassifier`` into our :class:`Tree`."""
    root = _dec_tree_to_my_tree_rec(0, dec_tree, features)
    return Tree(root, domains)


def _usefulness_count(tree: Tree, feature: str, domains: Dict[str, Tuple[float, float]],
                      clone: Callable[[Tree], Tree]) -> int:
    """Shared body of :func:`compute_score`; ``clone`` produces a working copy of
    ``tree``.  Passing ``Tree.clone`` gives the fast path; passing ``deepcopy``
    reproduces the original prototype (used only to benchmark the speedup)."""
    lo, hi = int(domains[feature][0]), int(domains[feature][1])

    # Conjunction over all values v of: (tree | feature = v).  An entity survives
    # iff the tree says True for *every* value of `feature` -> feature useless (True side).
    tree_all_true = clone(tree)
    tree_all_true.condition(feature, lo)
    for val in range(lo + 1, hi + 1):
        nxt = clone(tree)
        nxt.condition(feature, val)
        nxt.conjunction(tree_all_true)
        tree_all_true = nxt
    acum = tree_all_true.count()

    # Same, but for the negation: entities that are always False over `feature`.
    tree_all_false = clone(tree)
    tree_all_false.negate()
    tree_all_false.condition(feature, lo)
    for val in range(lo + 1, hi + 1):
        nxt = clone(tree)
        nxt.condition(feature, val)
        nxt.negate()
        nxt.conjunction(tree_all_false)
        tree_all_false = nxt
    acum += tree_all_false.count()

    # Total size of the (categorical) entity space.
    total = 1
    for f in domains:
        a, b = domains[f]
        total *= (math.floor(b) - math.ceil(a) + 1)

    return total - acum


def compute_score(tree: Tree, feature: str, domains: Dict[str, Tuple[float, float]]) -> int:
    """Usefulness score of ``feature``

    Counts the number of entities ``e`` for which ``feature`` is *useful*, i.e.
    for which some value change ``e -> e[feature=b]`` flips the prediction.  It
    equals ``|entity space| - #{e : prediction is constant over feature}``.

    Uses the fast copy-free scorer (``Tree.clone``); the result is identical to
    the ``copy.deepcopy`` prototype but from 3 to 5 times faster.
    """
    return _usefulness_count(tree, feature, domains, Tree.clone)


def usefulness_scores(clf: DecisionTreeClassifier,
                      domains: Dict[str, Tuple[float, float]],
                      features: Sequence[str]) -> np.ndarray:
    """Vector of usefulness scores aligned with ``features``."""
    tree = dec_tree_to_my_tree(clf, domains, features)
    return np.array([compute_score(tree, f, domains) for f in features], dtype=float)


# ---------------------------------------------------------------------------
# 2.  Datasets and a single, binning-strategy-aware trainer
# ---------------------------------------------------------------------------

def make_discretizer(n_bins: int, strategy: str, seed: int) -> KBinsDiscretizer:
    """Deterministic :class:`KBinsDiscretizer` that works across sklearn versions.

    We disable sub-sampling so that ``quantile``/``kmeans`` bin edges are fully
    reproducible, and silence the ``quantile_method`` FutureWarning on sklearn
    >= 1.5 by selecting an explicit method when the argument exists.
    """
    kwargs = dict(n_bins=n_bins, encode="ordinal", strategy=strategy)
    import inspect
    params = inspect.signature(KBinsDiscretizer.__init__).parameters
    if "subsample" in params:
        kwargs["subsample"] = None            # use every row -> deterministic edges
    if "random_state" in params:
        kwargs["random_state"] = seed         # only matters for the kmeans strategy
    if "quantile_method" in params:
        kwargs["quantile_method"] = "averaged_inverted_cdf"
    return KBinsDiscretizer(**kwargs)


# --- raw dataset loaders (cached at module level so we read each source once) ---

_CACHE: Dict[str, object] = {}


def load_california() -> Tuple[pd.DataFrame, np.ndarray]:
    if "california" not in _CACHE:
        housing = fetch_california_housing(as_frame=True)
        _CACHE["california"] = (housing.data.copy(), housing.target.to_numpy())
    X, y = _CACHE["california"]
    return X.copy(), y.copy()


def load_bike_sharing(csv_path: str = "datasets/bike_sharing/hour.csv") -> pd.DataFrame:
    if "bike" not in _CACHE:
        df = pd.read_csv(csv_path)
        df = df.drop(["casual", "registered", "instant", "dteday"], axis=1)
        _CACHE["bike"] = df
    return _CACHE["bike"].copy(deep=True)


def load_adult_income(url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
                      ) -> pd.DataFrame:
    """Load + ordinally-encode the Adult Income dataset (with an OpenML fallback)."""
    if "adult" in _CACHE:
        return _CACHE["adult"].copy(deep=True)

    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income",
    ]
    try:
        df = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)
    except Exception as exc:                       # pragma: no cover - network dependent
        warnings.warn(f"UCI download failed ({exc!r}); falling back to sklearn's OpenML copy.")
        from sklearn.datasets import fetch_openml
        adult = fetch_openml("adult", version=2, as_frame=True)
        df = adult.frame.rename(columns={"class": "income"})
        # openml uses '.'/'?' differently; align column names best-effort.
        df.columns = [c.replace("_", "-") for c in df.columns]

    # Drop rows with '?' in the usual columns, then any remaining NA.
    for col in ["workclass", "occupation", "native-country"]:
        if col in df.columns:
            df = df[df[col] != "?"]
    df = df.dropna().reset_index(drop=True)

    # Encode all columns as ordinal integers
    encoder = OrdinalEncoder()
    df[df.columns] = encoder.fit_transform(df[df.columns])
    _CACHE["adult"] = df
    return df.copy(deep=True)


@dataclass
class ModelBundle:
    """Everything downstream code needs about one trained model."""
    dataset: str
    clf: DecisionTreeClassifier
    accuracy: float
    domains: Dict[str, Tuple[float, float]]
    features: List[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    bins: int
    strategy: str


def _train_common(X: pd.DataFrame, y_binary: np.ndarray, domains: Dict[str, Tuple[float, float]],
                  features: List[str], dataset: str, bins: int, strategy: str,
                  seed: int, max_depth: int, leaves: int) -> ModelBundle:
    X = X[features]                                  # enforce a stable column order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.20, random_state=seed)
    clf = DecisionTreeClassifier(random_state=seed, max_depth=max_depth, max_leaf_nodes=leaves)
    clf.fit(X_train, y_train)
    acc = float((clf.predict(X_test) == y_test).mean())
    return ModelBundle(dataset, clf, acc, domains, features,
                       X_train.reset_index(drop=True), np.asarray(y_train),
                       X_test.reset_index(drop=True), np.asarray(y_test), bins, strategy)


def _densify(col: np.ndarray) -> Tuple[np.ndarray, int]:
    """Relabel an integer column to consecutive values 0..k-1 (order-preserving).

    ``KBinsDiscretizer`` can leave empty bins (skewed features under ``uniform``)
    or collapse bins (zero-inflated features under ``quantile``), so the raw
    ordinal output may skip values, e.g. ``{0,1,2,5}``.  Declaring the domain as
    ``(0, n_bins-1)`` would then count values that never occur ("phantom" values)
    and bias the score.  This relabels to the values actually present and returns
    the number of distinct bins ``k``; it does not change the model's partition of
    the data (the relabel is order-preserving), only the domain used by the score.
    """
    uniques, inverse = np.unique(np.asarray(col, dtype=int), return_inverse=True)
    return inverse.astype(int), len(uniques)


def train_tree(dataset: str, bins: int, strategy: str = "uniform", seed: int = 42,
               max_depth: int = 100_000, leaves: int = 1_000_000,
               correct_categorical_domains: bool = True) -> ModelBundle:
    """Train one decision tree for ``dataset`` with the requested discretization.

    Parameters
    ----------
    dataset : {"california", "bike_sharing", "adult_income"}
    bins : number of bins for the numeric features.
    strategy : {"uniform", "quantile", "kmeans"} passed to ``KBinsDiscretizer``.
    correct_categorical_domains : if True, the raw categorical Bike features
        ``season`` and ``weathersit`` use their true domain ``(1, 4)``.  Set to
        False to reproduce the original notebook (which used ``(0, 1)`` and
        ``(0, 4)`` respectively) for a like-for-like comparison.
    """
    kb = make_discretizer(bins, strategy, seed)

    if dataset == "california":
        X, y = load_california()
        cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude"]
        y_binary = (y > np.median(y)).astype(int)
        X_binned = pd.DataFrame(kb.fit_transform(X), columns=cols)
        domains = {}
        for c in cols:                                  # densify: domain = bins actually realized
            dense, k = _densify(X_binned[c].to_numpy())
            X_binned[c] = dense
            domains[c] = (0, k - 1)
        return _train_common(X_binned, y_binary, domains, cols, dataset, bins, strategy, seed, max_depth, leaves)

    if dataset == "bike_sharing":
        df = load_bike_sharing()
        y = df["cnt"].to_numpy()
        y_binary = (y > np.median(y)).astype(int)
        to_bin = ["mnth", "hr", "weekday", "temp", "atemp", "hum", "windspeed"]
        binned = kb.fit_transform(df[to_bin])
        domains = {}
        for i, c in enumerate(to_bin):
            dense, k = _densify(binned[:, i])           # densify: domain = bins actually realized
            df[c + "_binned"] = dense
            domains[c + "_binned"] = (0, k - 1)
        X = df.drop(columns=to_bin + ["cnt"])
        features = list(X.columns)
        # Raw (un-binned) categorical features keep their true integer domains.
        domains["yr"] = (0, 1)
        domains["holiday"] = (0, 1)
        domains["workingday"] = (0, 1)
        if correct_categorical_domains:
            domains["season"] = (1, 4)
            domains["weathersit"] = (1, 4)
        else:
            domains["season"] = (0, 1)          # original (buggy) values
            domains["weathersit"] = (0, 4)
        return _train_common(X, y_binary, domains, features, dataset, bins, strategy, seed, max_depth, leaves)

    if dataset == "adult_income":
        df = load_adult_income()
        y = df["income"].to_numpy().astype(int)
        to_bin = ["age", "fnlwgt", "workclass", "education", "education-num", "marital-status",
                  "occupation", "relationship", "capital-gain", "capital-loss",
                  "hours-per-week", "native-country"]
        binned = kb.fit_transform(df[to_bin])
        domains = {}
        for i, c in enumerate(to_bin):
            dense, k = _densify(binned[:, i])           # densify: domain = bins actually realized
            df[c + "_binned"] = dense
            domains[c + "_binned"] = (0, k - 1)
        X = df.drop(columns=to_bin + ["income"])
        features = list(X.columns)
        domains["race"] = (0, 4)
        domains["sex"] = (0, 1)
        return _train_common(X, y, domains, features, dataset, bins, strategy, seed, max_depth, leaves)

    raise ValueError(f"Unknown dataset {dataset!r}")


# ---------------------------------------------------------------------------
# 3.  Importance methods to compare against the usefulness score
# ---------------------------------------------------------------------------
#
# Every method returns a non-negative importance vector aligned with
# ``bundle.features``.  We deliberately include a diverse set:
#   * usefulness  - our logic-based score (exact, on the tree itself);
#   * permutation - model-agnostic, measured on held-out data;
#   * shap        - TreeSHAP, the baseline already used in the paper;
#   * lime        - local surrogates, aggregated to a global score.


def importance_permutation(bundle: ModelBundle, n_repeats: int = 10, seed: int = 42) -> np.ndarray:
    """Permutation importance, measured on the held-out test split."""
    res = permutation_importance(bundle.clf, bundle.X_test, bundle.y_test,
                                 n_repeats=n_repeats, random_state=seed, scoring="accuracy")
    # Negative means "removing the feature helped"; clip so importances stay >= 0.
    return np.clip(res.importances_mean, 0.0, None)


def importance_shap(bundle: ModelBundle, sample: int = 200, seed: int = 42) -> np.ndarray:
    """Global TreeSHAP importance: mean over instances of the absolute SHAP value.

    Multi-class contributions are summed (as in the paper's appendix).
    """
    import shap
    X = bundle.X_train
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(bundle.clf)
    sv = explainer.shap_values(Xs, check_additivity=False)

    if isinstance(sv, list):                          # list per class -> sum |.|
        contrib = np.sum([np.abs(np.asarray(s)) for s in sv], axis=0)
    else:
        sv = np.asarray(sv)
        contrib = np.sum(np.abs(sv), axis=2) if sv.ndim == 3 else np.abs(sv)
    return contrib.mean(axis=0)


def importance_lime(bundle: ModelBundle, sample: int = 100, seed: int = 42) -> np.ndarray:
    """Global LIME importance: mean absolute local weight over a sample of rows.

    All features are treated as categorical (they are ordinal bins), so LIME's
    continuous discretizer is switched off.
    """
    from lime.lime_tabular import LimeTabularExplainer
    X = bundle.X_train.to_numpy(dtype=float)
    d = X.shape[1]
    explainer = LimeTabularExplainer(
        X, mode="classification", feature_names=list(bundle.features),
        categorical_features=list(range(d)), discretize_continuous=False,
        random_state=seed)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)

    agg = np.zeros(d)
    label = int(bundle.clf.classes_[-1])
    for i in idx:
        exp = explainer.explain_instance(X[i], bundle.clf.predict_proba,
                                         num_features=d, labels=(label,))
        for fidx, w in exp.as_map()[label]:
            agg[fidx] += abs(w)
    return agg / len(idx)


# Registry of every importance method (name -> callable(bundle, cfg)).
def compute_all_importances(bundle: ModelBundle, cfg: Config,
                            methods: Sequence[str] = ("usefulness", "permutation",
                                                      "shap", "lime"),
                            ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute the requested importance vectors, timing each one.

    Returns ``(importances, timings_seconds)``.  Methods whose optional
    dependency is missing are skipped with a warning rather than crashing.
    """
    importances: Dict[str, np.ndarray] = {}
    timings: Dict[str, float] = {}

    for m in methods:
        t0 = time.perf_counter()
        try:
            if m == "usefulness":
                importances[m] = usefulness_scores(bundle.clf, bundle.domains, bundle.features)
            elif m == "permutation":
                importances[m] = importance_permutation(bundle, seed=cfg.seed)
            elif m == "shap":
                importances[m] = importance_shap(bundle, sample=cfg.importance_sample, seed=cfg.seed)
            elif m == "lime":
                importances[m] = importance_lime(bundle, sample=min(cfg.importance_sample, 100), seed=cfg.seed)
            else:
                raise ValueError(f"Unknown importance method {m!r}")
            timings[m] = time.perf_counter() - t0
        except Exception as exc:                      # optional dep missing / method failed
            warnings.warn(f"Importance method {m!r} skipped: {exc!r}")
    return importances, timings


def normalize_importance(v: np.ndarray, how: str = "sum") -> np.ndarray:
    """Scale an importance vector so different methods can be plotted together."""
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    if how == "sum":
        s = v.sum()
        return v / s if s > 0 else v
    if how == "max":
        m = v.max()
        return v / m if m > 0 else v
    raise ValueError(how)


# ---------------------------------------------------------------------------
# 4.  Comparing rankings: correlations, top-k overlap, rank-biased overlap
# ---------------------------------------------------------------------------
#
# Because the methods live on different scales, we compare the *rankings*
# they induce, not the raw numbers.  Three complementary views:
#   * rank correlation (Spearman / Kendall) between every pair of methods --
#     treats every position in the ranking as equally important;
#   * top-k overlap of each method with the usefulness score -- this is exactly
#     the metric of Table 1 in the paper, here extended to every method;
#   * Rank-Biased Overlap (RBO) between every pair of methods -- unlike
#     Spearman/Kendall, RBO is *top-weighted*, so it rewards two methods for
#     agreeing on the most important features even if they disagree further
#     down the ranking.


def ranking_from_importance(imp: np.ndarray, features: Sequence[str]) -> List[str]:
    """Return features ordered from most to least important."""
    return [features[i] for i in np.argsort(-np.asarray(imp, dtype=float))]


def rank_correlation_matrix(importances: Dict[str, np.ndarray], method: str = "spearman") -> pd.DataFrame:
    """Pairwise Spearman (or Kendall) correlation between the methods' importances."""
    corr = spearmanr if method == "spearman" else kendalltau
    names = list(importances)
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            r = corr(importances[a], importances[b]).correlation
            M.loc[a, b] = M.loc[b, a] = r
    return M


def topk_intersection_vs(importances: Dict[str, np.ndarray], features: Sequence[str],
                         reference: str = "usefulness", ks: Sequence[int] = (1, 3, 5, 7)
                         ) -> pd.DataFrame:
    """|top-k(method) ∩ top-k(reference)| for each method and k (paper's Table 1)."""
    ref_rank = ranking_from_importance(importances[reference], features)
    rows = {}
    for name, imp in importances.items():
        r = ranking_from_importance(imp, features)
        rows[name] = {f"top-{k}": len(set(r[:k]) & set(ref_rank[:k])) for k in ks}
    return pd.DataFrame(rows).T


def rank_biased_overlap(rank_a: Sequence[str], rank_b: Sequence[str], p: float = 0.9) -> float:
    """Rank-Biased Overlap (Webber, Moffat & Zobel, 2010) between two complete
    rankings of the same items.

    Unlike Spearman/Kendall, RBO is *top-weighted*: agreement near the top of
    the ranking counts for more than agreement near the bottom, which matches
    how a ranking of feature importances is actually read.  ``p`` sets how
    quickly that weight decays with depth: closer to 0 -> only the very top
    matters; closer to 1 -> the whole ranking matters almost equally.  Returns
    1.0 for identical rankings.
    """
    k = len(rank_a)
    if k == 0:
        return 1.0
    set_a, set_b = set(), set()
    overlap = 0
    weighted_sum = 0.0
    for d in range(1, k + 1):
        set_a.add(rank_a[d - 1])
        set_b.add(rank_b[d - 1])
        overlap = len(set_a & set_b)
        weighted_sum += (overlap / d) * (p ** d)
    x_k = overlap / k                                   # = 1.0 here: both rank the same feature set
    return (1 - p) / p * weighted_sum + x_k * p ** k


def rank_biased_overlap_matrix(importances: Dict[str, np.ndarray], features: Sequence[str],
                               p: float = 0.9) -> pd.DataFrame:
    """Pairwise Rank-Biased Overlap between the rankings the methods induce."""
    names = list(importances)
    rankings = {m: ranking_from_importance(importances[m], features) for m in names}
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            r = rank_biased_overlap(rankings[a], rankings[b], p=p)
            M.loc[a, b] = M.loc[b, a] = r
    return M


# ---------------------------------------------------------------------------
# 5.  Runtime analysis
# ---------------------------------------------------------------------------
#
# We measure wall-clock time with ``time.perf_counter`` (best-of-``repeat`` to
# damp noise), from the standard library, so the measurement itself adds no
# dependency and is fully reproducible.  The two natural "size" axes for the
# usefulness algorithm are the number of tree nodes and the per-feature domain
# size (number of bins), so those are what we sweep and plot against.


def entity_space_log10(domains: Dict[str, Tuple[float, float]]) -> float:
    """log10 of the categorical entity-space size (it easily overflows float)."""
    return float(sum(math.log10(math.floor(b) - math.ceil(a) + 1) for a, b in domains.values()))


def profile_usefulness(bundle: ModelBundle, repeat: int = 5) -> Dict[str, float]:
    """Time of computing *all* usefulness scores for one model (best of `repeat`)."""
    tree = dec_tree_to_my_tree(bundle.clf, bundle.domains, bundle.features)

    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for f in bundle.features:
            compute_score(tree, f, bundle.domains)
        best = min(best, time.perf_counter() - t0)

    return {
        "dataset": bundle.dataset, "bins": bundle.bins, "strategy": bundle.strategy,
        "n_features": len(bundle.features), "n_nodes": int(bundle.clf.tree_.node_count),
        "max_depth": int(bundle.clf.get_depth()), "accuracy": bundle.accuracy,
        "entity_space_log10": entity_space_log10(bundle.domains),
        "time_total_s": best, "time_per_feature_s": best / len(bundle.features),
    }


def scaling_experiment(dataset: str, cfg: Config, strategy: str = "uniform",
                       bins_list: Optional[Sequence[int]] = None,
                       seed: Optional[int] = None) -> pd.DataFrame:
    """Sweep #bins and profile the usefulness algorithm's own runtime.

    Varies ``bins`` at a fixed, generous leaf budget, isolating how the
    usefulness score's cost grows with the categorical entity-space size.
    (For how *every* method's runtime scales with tree size instead, see
    :func:`method_scaling_experiment`.)
    """
    seed = cfg.seed if seed is None else seed
    bins_list = list(cfg.bins_grid) if bins_list is None else list(bins_list)
    rows = []
    for bins in bins_list:
        b = train_tree(dataset, bins=bins, strategy=strategy, seed=seed, leaves=100 * bins)
        rows.append(profile_usefulness(b))
    return pd.DataFrame(rows)


def method_scaling_experiment(dataset: str, cfg: Config, strategy: str = "uniform",
                              bins: Optional[int] = None,
                              leaves_list: Optional[Sequence[int]] = None,
                              methods: Sequence[str] = ("usefulness", "permutation",
                                                        "shap", "lime"),
                              repeat: int = 10, seed: Optional[int] = None) -> pd.DataFrame:
    """Wall-clock of *every* importance method vs tree size (one line per method).

    Bins are held fixed (middle of ``cfg.bins_grid`` by default) while
    ``max_leaf_nodes`` is swept over a fine, log-spaced grid (14 points from 25
    to 6400 leaves by default -- denser than :func:`scaling_experiment`'s
    tree-size sweep used to be).  Each (method, tree size) pair is timed
    ``repeat`` times, so the run-to-run measurement noise (mean +/- std) is
    visible, rather than a single best-of estimate.
    """
    seed = cfg.seed if seed is None else seed
    bins = cfg.bins_grid[len(cfg.bins_grid) // 2] if bins is None else bins
    if leaves_list is None:
        leaves_list = [int(round(x)) for x in np.geomspace(25, 6400, 14)]

    rows = []
    warmed_up = set()
    for leaves in leaves_list:
        b = train_tree(dataset, bins=bins, strategy=strategy, seed=seed, leaves=leaves)
        n_nodes = int(b.clf.tree_.node_count)
        for m in methods:
            if m not in warmed_up:                      # untimed warm-up (e.g. SHAP numba JIT)
                try:
                    compute_all_importances(b, cfg, methods=(m,))
                except Exception:
                    pass
                warmed_up.add(m)
            times = []
            for _ in range(repeat):
                try:
                    _, timings = compute_all_importances(b, cfg, methods=(m,))
                except Exception as exc:
                    warnings.warn(f"{m} skipped at leaves={leaves}: {exc!r}")
                    break
                if m in timings:
                    times.append(timings[m])
            if times:
                arr = np.asarray(times)
                rows.append({
                    "dataset": dataset, "bins": bins, "leaves": leaves, "n_nodes": n_nodes,
                    "method": m, "time_mean_s": float(arr.mean()), "time_std_s": float(arr.std()),
                    "time_min_s": float(arr.min()), "time_max_s": float(arr.max()),
                    "n_repeats": len(times),
                })
    return pd.DataFrame(rows)


def method_runtime_comparison(bundle: ModelBundle, cfg: Config,
                              methods: Sequence[str] = ("usefulness", "permutation",
                                                        "shap", "lime"),
                              repeat: int = 3) -> pd.DataFrame:
    """Wall-clock cost of each importance method on the *same* trained model.

    This is the head-to-head "performance analysis" the reviewers asked for and
    puts the paper's "extremely fast" claim on a quantitative footing.
    """
    rows = []
    for m in methods:
        best = math.inf
        ok = True
        for _ in range(repeat):
            imp_one, timings = compute_all_importances(bundle, cfg, methods=(m,))
            if m not in timings:                       # method skipped (missing dep)
                ok = False
                break
            best = min(best, timings[m])
        if ok:
            rows.append({"method": m, "time_s": best,
                         "n_nodes": int(bundle.clf.tree_.node_count),
                         "n_features": len(bundle.features)})
    return pd.DataFrame(rows)


# --- implementation speedup: copy-free scorer vs copy.deepcopy -------------
def profile_scorer_speedup(bundle: ModelBundle, repeat: int = 3) -> Dict[str, float]:
    """Quantify how much of the usefulness runtime was ``copy.deepcopy`` overhead.

    Times the *same algorithm* with the fast ``Tree.clone`` and with the original
    ``copy.deepcopy``.  The results are identical; only the copy strategy differs,
    so this isolates the prototype's overhead from the algorithm's real cost.
    """
    tree = dec_tree_to_my_tree(bundle.clf, bundle.domains, bundle.features)

    def _bench(clone) -> float:
        best = math.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            for f in bundle.features:
                _usefulness_count(tree, f, bundle.domains, clone)
            best = min(best, time.perf_counter() - t0)
        return best

    slow, fast = _bench(deepcopy), _bench(Tree.clone)
    return {"dataset": bundle.dataset, "n_nodes": int(bundle.clf.tree_.node_count),
            "deepcopy_s": slow, "clone_s": fast, "speedup": slow / fast}


# ---------------------------------------------------------------------------
# 6.  Binning-strategy sensitivity analysis
# ---------------------------------------------------------------------------
#
# We re-run the usefulness experiment under ``uniform`` / ``quantile`` / ``kmeans``
# discretization and ask two questions:
#   * does model accuracy depend on the strategy?
#   * does the *ranking* the usefulness score induces depend on the strategy?
#     (measured by cross-strategy Spearman correlation and top-3 overlap);
# This directly quantifies how much the paper's discretization choice matters.


def mean_usefulness_importance(dataset: str, bins: int, strategy: str,
                               n_models: int, seed: int, leaves: int) -> Tuple[np.ndarray, List[str], float]:
    """Average (sum-normalized) usefulness importance over ``n_models`` trees.

    Averaging the normalized vectors makes the aggregate scale-invariant, so
    models with larger entity spaces do not dominate the mean.
    """
    rng = random.Random(seed)
    acc_sum = 0.0
    agg = None
    features = None
    for _ in range(n_models):
        s = rng.randint(0, 100_000)
        b = train_tree(dataset, bins=bins, strategy=strategy, seed=s, leaves=leaves)
        features = b.features
        v = normalize_importance(usefulness_scores(b.clf, b.domains, b.features))
        agg = v if agg is None else agg + v
        acc_sum += b.accuracy
    return agg / n_models, list(features), acc_sum / n_models


def run_binning_experiment(dataset: str, cfg: Config, n_models: Optional[int] = None,
                           bins_grid: Optional[Sequence[int]] = None,
                           strategies: Optional[Sequence[str]] = None,
                           seed: Optional[int] = None
                           ) -> Dict[str, pd.DataFrame]:
    """Full binning sensitivity run, returning four tidy tables.

    Returns ``{"summary", "stability", "bins_stability", "importance"}``
    DataFrames, suitable both for the plots in :func:`plot_binning_sensitivity`
    and for saving to CSV.  ``stability`` compares strategies at each fixed
    bin count; ``bins_stability`` compares bin counts at each fixed strategy --
    both report Spearman, Rank-Biased Overlap (``cfg.rbo_p``), and top-3 overlap.
    """
    n_models = cfg.n_models if n_models is None else n_models
    bins_grid = list(cfg.bins_grid) if bins_grid is None else list(bins_grid)
    strategies = list(cfg.strategies) if strategies is None else list(strategies)
    seed = cfg.seed if seed is None else seed
    leaves_per_bin = cfg.leaves_per_bin[dataset]

    # mean importance vector per (strategy, bins)
    mean_imp: Dict[Tuple[str, int], np.ndarray] = {}
    features_ref: List[str] = []
    summary_rows, importance_rows = [], []

    for strategy in strategies:
        for bins in bins_grid:
            v, features, acc = mean_usefulness_importance(
                dataset, bins, strategy, n_models, seed, leaves=leaves_per_bin * bins)
            mean_imp[(strategy, bins)] = v
            features_ref = features
            summary_rows.append({"dataset": dataset, "strategy": strategy, "bins": bins,
                                 "accuracy": acc})
            for f, val in zip(features, v):
                importance_rows.append({"strategy": strategy, "bins": bins,
                                        "feature": f.replace("_binned", ""), "importance": val})

    # cross-strategy stability at each bin count (does the ranking depend on
    # *how* we discretize, holding the number of bins fixed?)
    stability_rows = []
    for bins in bins_grid:
        for i, sa in enumerate(strategies):
            for sb in strategies[i + 1:]:
                va, vb = mean_imp[(sa, bins)], mean_imp[(sb, bins)]
                ra = ranking_from_importance(va, features_ref)
                rb = ranking_from_importance(vb, features_ref)
                stability_rows.append({
                    "dataset": dataset, "bins": bins, "strategy_a": sa, "strategy_b": sb,
                    "spearman": spearmanr(va, vb).correlation,
                    "rbo": rank_biased_overlap(ra, rb, p=cfg.rbo_p),
                    "top3_overlap": len(set(ra[:3]) & set(rb[:3])),
                })

    # cross-bins-count stability at each strategy (does the ranking depend on
    # *how many* bins we use, holding the discretization strategy fixed?)
    bins_stability_rows = []
    for strategy in strategies:
        for i, ba in enumerate(bins_grid):
            for bb in bins_grid[i + 1:]:
                va, vb = mean_imp[(strategy, ba)], mean_imp[(strategy, bb)]
                ra = ranking_from_importance(va, features_ref)
                rb = ranking_from_importance(vb, features_ref)
                bins_stability_rows.append({
                    "dataset": dataset, "strategy": strategy, "bins_a": ba, "bins_b": bb,
                    "spearman": spearmanr(va, vb).correlation,
                    "rbo": rank_biased_overlap(ra, rb, p=cfg.rbo_p),
                    "top3_overlap": len(set(ra[:3]) & set(rb[:3])),
                })

    return {"summary": pd.DataFrame(summary_rows),
            "stability": pd.DataFrame(stability_rows),
            "bins_stability": pd.DataFrame(bins_stability_rows),
            "importance": pd.DataFrame(importance_rows)}


# ---------------------------------------------------------------------------
# 6b.  The paper's main experiment: usefulness scores
# ---------------------------------------------------------------------------
#
# For each bin count we train # ``n_models`` trees and aggregate each feature's 
# usefulness score into its # Q1 / median / Q3, plus the mean model accuracy.


def run_score_experiment(dataset: str, cfg: Config, strategy: str = "uniform",
                         n_models: Optional[int] = None, bins_grid: Optional[Sequence[int]] = None,
                         seed: Optional[int] = None) -> Dict[str, object]:
    """Compute the usefulness-score summary per bin count.

    Returns ``{"per_bins": {bins: {...}}, "table": DataFrame}``.  The default
    ``uniform`` strategy matches the paper text.
    """
    n_models = cfg.n_models if n_models is None else n_models
    bins_grid = list(cfg.bins_grid) if bins_grid is None else list(bins_grid)
    seed = cfg.seed if seed is None else seed
    leaves_per_bin = cfg.leaves_per_bin[dataset]

    per_bins, rows = {}, []
    for bins in bins_grid:
        rng = random.Random(seed)                     # same seed sequence per bin count -> reproducible
        per_feature: Dict[str, List[float]] = {}
        features: List[str] = []
        accuracies = []
        for _ in range(n_models):
            s = rng.randint(0, 100_000)
            b = train_tree(dataset, bins=bins, strategy=strategy, seed=s, leaves=leaves_per_bin * bins)
            features = b.features
            for f, val in zip(b.features, usefulness_scores(b.clf, b.domains, b.features)):
                per_feature.setdefault(f, []).append(val)
            accuracies.append(b.accuracy)

        # aggregate to quartiles and sort by median (as in the paper figures)
        stats = [(f.replace("_binned", ""), np.percentile(per_feature[f], 25),
                  float(np.median(per_feature[f])), np.percentile(per_feature[f], 75))
                 for f in features]
        stats.sort(key=lambda t: t[2])
        mean_acc = float(np.mean(accuracies))
        per_bins[bins] = {
            "labels": [s[0] for s in stats], "q1": [s[1] for s in stats],
            "median": [s[2] for s in stats], "q3": [s[3] for s in stats],
            "mean_accuracy": mean_acc,
        }
        for name, q1, med, q3 in stats:
            rows.append({"dataset": dataset, "strategy": strategy, "bins": bins,
                         "feature": name, "q1": q1, "median": med, "q3": q3,
                         "mean_accuracy": mean_acc})

    return {"per_bins": per_bins, "table": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# 7a.  Orchestration for the multi-method comparison
# ---------------------------------------------------------------------------


def _mean_of_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Element-wise mean of a list of identically-shaped/labelled DataFrames."""
    acc = frames[0].copy() * 0.0
    for fr in frames:
        acc = acc + fr
    return acc / len(frames)


def run_comparison_experiment(dataset: str, cfg: Config, bins: int = 6,
                              methods: Sequence[str] = ("usefulness", "permutation",
                                                        "shap", "lime"),
                              n_models: Optional[int] = None, seed: Optional[int] = None
                              ) -> Dict[str, object]:
    """Compare the usefulness ranking with every method over ``n_models`` trees.

    Aggregates (like the paper's Table 1) across models and returns:
      * ``topk``          - mean top-k overlap of each method with usefulness;
      * ``corr_spearman`` - mean pairwise Spearman rank-correlation matrix;
      * ``rbo``           - mean pairwise Rank-Biased Overlap matrix (top-weighted;
                            persistence ``cfg.rbo_p``);
      * ``mean_importance`` - mean normalized importance per (feature, method);
      * ``features``      - feature order (by mean usefulness), for plotting.
    """
    n_models = cfg.n_models if n_models is None else n_models
    seed = cfg.seed if seed is None else seed
    leaves = cfg.leaves_per_bin[dataset] * bins
    rng = random.Random(seed)

    topk_frames, corr_frames, rbo_frames = [], [], []
    imp_accum: Dict[str, np.ndarray] = {}
    features: List[str] = []

    for _ in range(n_models):
        s = rng.randint(0, 100_000)
        b = train_tree(dataset, bins=bins, strategy="uniform", seed=s, leaves=leaves)
        features = b.features
        imp, _ = compute_all_importances(b, cfg, methods=methods)
        present = [m for m in methods if m in imp]

        topk_frames.append(topk_intersection_vs(imp, features).loc[present])
        rbo_frames.append(rank_biased_overlap_matrix({m: imp[m] for m in present}, features, p=cfg.rbo_p))
        corr_frames.append(rank_correlation_matrix({m: imp[m] for m in present}))
        for m in present:
            nv = normalize_importance(imp[m])
            imp_accum[m] = nv if m not in imp_accum else imp_accum[m] + nv

    mean_importance = pd.DataFrame(
        {m: imp_accum[m] / n_models for m in imp_accum},
        index=[f.replace("_binned", "") for f in features])

    return {
        "dataset": dataset, "bins": bins, "n_models": n_models,
        "topk": _mean_of_frames(topk_frames),
        "corr_spearman": _mean_of_frames(corr_frames),
        "rbo": _mean_of_frames(rbo_frames),
        "mean_importance": mean_importance,
        "features": [f.replace("_binned", "") for f in features],
    }


# ---------------------------------------------------------------------------
# 7b.  Plotting  (all figures saved as PNG)
# ---------------------------------------------------------------------------


def _save(fig, path_no_ext: str) -> None:
    fig.savefig(path_no_ext + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_method_importance_heatmap(comparison: Dict[str, object], path_no_ext: str) -> None:
    """Compact overview: normalized importance, features (sorted) x methods."""
    df = comparison["mean_importance"].copy()
    df = df.reindex(df["usefulness"].sort_values(ascending=False).index)  # sort by usefulness
    # column-normalize so each method's color spans [0, 1] (relative within method)
    disp = df / df.max(axis=0)
    fig, ax = plt.subplots(figsize=(1.4 * df.shape[1] + 2, 0.42 * df.shape[0] + 2))
    im = ax.imshow(disp.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=0, ha="center")
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ds = PRETTY_NAMES.get(comparison['dataset'], comparison['dataset'])
    ax.set_title(f"Normalized feature importance - {ds} ({comparison['bins']} bins)")
    for i in range(df.shape[0]):                       # annotate cells (text stays ink-colored)
        for j in range(df.shape[1]):
            ax.text(j, i, f"{disp.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="#111" if disp.values[i, j] < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="importance (normalized)")
    _save(fig, path_no_ext)


def plot_rank_correlation_heatmap(comparison: Dict[str, object], path_no_ext: str) -> None:
    """Mean pairwise Spearman correlation between methods (diverging, centred 0)."""
    import matplotlib.colors as mcolors
    M = comparison["corr_spearman"]
    fig, ax = plt.subplots(figsize=(0.9 * len(M) + 2, 0.9 * len(M) + 1.5))
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)          # neutral grey at 0
    im = ax.imshow(M.values, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index, rotation=0, ha="right")
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#111" if abs(M.values[i, j]) < 0.6 else "white")
    ds = PRETTY_NAMES.get(comparison['dataset'], comparison['dataset'])
    ax.set_title(f"Rank correlation - {ds}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Spearman $\rho$")
    _save(fig, path_no_ext)


def plot_rank_biased_overlap_heatmap(comparison: Dict[str, object], path_no_ext: str) -> None:
    """Mean pairwise Rank-Biased Overlap between methods (top-weighted; unlike
    Spearman/Kendall, RBO lives in [0, 1], so this uses a sequential colormap
    rather than a diverging one."""
    M = comparison["rbo"]
    fig, ax = plt.subplots(figsize=(0.9 * len(M) + 2, 0.9 * len(M) + 1.5))
    im = ax.imshow(M.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index, rotation=0, ha="right")
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#111" if M.values[i, j] > 0.6 else "white")
    ds = PRETTY_NAMES.get(comparison['dataset'], comparison['dataset'])
    ax.set_title(f"Rank-Biased Overlap - {ds}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RBO")
    _save(fig, path_no_ext)


def plot_topk_intersection(comparison: Dict[str, object], path_no_ext: str) -> None:
    """Top-k overlap of each method with the usefulness ranking (extends Table 1)."""
    df = comparison["topk"].drop(index=["usefulness"], errors="ignore")
    ks = list(df.columns)
    methods = list(df.index)
    x = np.arange(len(methods))
    w = 0.8 / len(ks)
    fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2, 4.5))
    for i, k in enumerate(ks):
        ax.bar(x + i * w - 0.4 + w / 2, df[k].values, width=w, label=k, color=COLOR_SCHEME[i])
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=0, ha="center")
    ax.set_ylabel("# shared features (top-k)")
    ds = PRETTY_NAMES.get(comparison['dataset'], comparison['dataset'])
    ax.set_title(f"Top-k overlap with usefulness - {ds}")
    ax.legend(title="", ncol=len(ks))
    _save(fig, path_no_ext)


def plot_scaling(scaling_df: pd.DataFrame, dataset: str, path_no_ext: str) -> None:
    """Usefulness-score runtime vs number of bins (entity-space size).

    (For runtime vs tree size, across every method, see
    :func:`plot_method_scaling` instead.)
    """
    df = scaling_df.sort_values("bins")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(df["bins"], df["time_total_s"], "o-", color=COLOR_SCHEME[0])
    ax.set_yscale("log"); ax.set_xlabel("number of bins")
    ax.set_ylabel("time, all features (s)"); ax.set_title("Time vs number of bins")

    fig.suptitle(f"Usefulness-score scaling - {PRETTY_NAMES.get(dataset, dataset)}", fontweight="bold")
    fig.tight_layout()
    _save(fig, path_no_ext)


def plot_method_scaling(df: pd.DataFrame, dataset: str, path_no_ext: str) -> None:
    """Runtime vs tree size for every method in ``df`` (log-log, mean line + std
    band), plus a companion bar chart of each method's timing variance
    (coefficient of variation, averaged across tree sizes).

    Saved as two separate images -- ``{path_no_ext}_runtime.png`` and
    ``{path_no_ext}_variance.png`` -- so either can be dropped into a paper on
    its own.  Pass a pre-filtered ``df`` (e.g. only the "usefulness"/"shap"
    rows) for a focused head-to-head of just those methods.
    """
    methods_present = list(df["method"].unique())

    # time vs tree size, one line + std band per method
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods_present:
        d = df[df["method"] == m].sort_values("n_nodes")
        color = METHOD_COLORS.get(m, COLOR_SCHEME[7])
        ax.plot(d["n_nodes"], d["time_mean_s"], "o-", color=color, label=m)
        lo = np.clip(d["time_mean_s"] - d["time_std_s"], 1e-6, None)
        hi = d["time_mean_s"] + d["time_std_s"]
        ax.fill_between(d["n_nodes"], lo, hi, color=color, alpha=0.15, linewidth=0)
    if "usefulness" in methods_present:                # power-law fit, as a complexity check
        d = df[df["method"] == "usefulness"].sort_values("n_nodes")
        x, y = d["n_nodes"].to_numpy(float), d["time_mean_s"].to_numpy(float)
        if len(x) >= 2 and (x > 0).all() and (y > 0).all():
            slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, np.exp(intercept) * xs ** slope, "--",
                    color=METHOD_COLORS.get("usefulness", "black"), linewidth=1,
                    label=fr"usefulness fit $\propto |T|^{{{slope:.2f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of tree nodes"); ax.set_ylabel("time, all features (s)")
    ax.set_title(f"Runtime vs tree size, per method - {PRETTY_NAMES.get(dataset, dataset)}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, f"{path_no_ext}_runtime")

    # timing variance per method (coefficient of variation, averaged over sizes)
    fig, ax = plt.subplots(figsize=(1.6 * len(methods_present) + 2, 4.5))
    cv = (df["time_std_s"] / df["time_mean_s"]).groupby(df["method"]).mean()
    order = [m for m in ("usefulness", "permutation", "shap", "lime") if m in cv.index]
    cv = cv.reindex(order)
    colors = [METHOD_COLORS.get(m, COLOR_SCHEME[7]) for m in order]
    ax.bar(order, cv.values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("coeff. of variation (std / mean)")
    n_reps = int(df["n_repeats"].iloc[0]) if len(df) else 0
    ax.set_title(f"Timing variance across tree sizes - {PRETTY_NAMES.get(dataset, dataset)}\n({n_reps} reps/size)")
    fig.tight_layout()
    _save(fig, f"{path_no_ext}_variance")


# ---------------------------------------------------------------------------
# 8.  Unit tests for the tree algebra
# ---------------------------------------------------------------------------


def run_tree_tests() -> None:
    """Assert the known model counts of the small reference tree (raises on error)."""
    false = Node("False", 0, None, None)
    true = Node("True", 1, None, None)
    x1 = Node("x", 1.5, deepcopy(true), deepcopy(false))
    x2 = Node("x", 0.5, deepcopy(true), x1)
    y = Node("y", 0.5, deepcopy(false), deepcopy(true))
    z = Node("z", 0.5, x2, y)
    domains = {"x": (0, 2), "y": (0, 1), "z": (0, 1)}
    tree = Tree(z, domains)

    assert tree.count() == 7, tree.count()

    neg = deepcopy(tree); neg.negate()
    assert tree.count() + neg.count() == 12                     # partition of the 12 entities

    a = deepcopy(tree); b = deepcopy(tree); b.negate(); a.conjunction(b)
    assert a.count() == 0                                       # φ ∧ ¬φ is unsatisfiable

    for feat, val, exp in [("z", 0, 8), ("z", 1, 6), ("x", 1, 9)]:
        c = deepcopy(tree); c.condition(feat, val)
        assert c.count() == exp, (feat, val, c.count())

    c = deepcopy(tree); c.condition("x", 1); c.condition("y", 1)
    assert c.count() == 12

    # FIX-3: a feature whose domain starts at 1 must be scored correctly.
    s = Node("s", 2.5, Node("True", 1, None, None), Node("False", 0, None, None))
    assert compute_score(Tree(s, {"s": (1, 4)}), "s", {"s": (1, 4)}) == 4
    print("run_tree_tests: all assertions passed")


if __name__ == "__main__":       # tiny smoke test (offline: uses only Bike Sharing)
    set_all_seeds(42)
    set_plot_style()
    run_tree_tests()
    bundle = train_tree("bike_sharing", bins=4, strategy="uniform", seed=42, leaves=400)
    scores = usefulness_scores(bundle.clf, bundle.domains, bundle.features)
    ranked = sorted(zip(bundle.features, scores), key=lambda t: -t[1])
    print(f"Bike Sharing (4 bins): acc={bundle.accuracy:.3f}, "
          f"top feature = {ranked[0][0]}")
    print(profile_usefulness(bundle))


def plot_method_runtimes(runtime_df: pd.DataFrame, dataset: str, path_no_ext: str) -> None:
    """Per-method wall-clock cost on one model (log y -costs span decades)."""
    df = runtime_df.sort_values("time_s")
    colors = [METHOD_COLORS.get(m, COLOR_SCHEME[7]) for m in df["method"]]
    fig, ax = plt.subplots(figsize=(1.3 * len(df) + 2, 4.5))
    ax.bar(df["method"], df["time_s"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("time (s, log scale)")
    ax.set_title(f"Runtime per importance method - {PRETTY_NAMES.get(dataset, dataset)}\n"
                 f"(tree with {int(df['n_nodes'].iloc[0])} nodes, {int(df['n_features'].iloc[0])} features)")
    for i, (m, t) in enumerate(zip(df["method"], df["time_s"])):
        ax.text(i, t, f"{t:.3g}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticklabels(df["method"], rotation=20, ha="right")
    _save(fig, path_no_ext)


def plot_scorer_speedup(speedups: pd.DataFrame, path_no_ext: str) -> None:
    """Grouped bars: deepcopy prototype vs copy-free scorer, per dataset (log y)."""
    fig, ax = plt.subplots(figsize=(1.6 * len(speedups) + 2, 4.5))
    x = np.arange(len(speedups))
    ax.bar(x - 0.2, speedups["deepcopy_s"], width=0.4, color=COLOR_SCHEME[3], label="copy.deepcopy (prototype)")
    ax.bar(x + 0.2, speedups["clone_s"], width=0.4, color=COLOR_SCHEME[2], label="copy-free clone")
    for xi, (_, r) in zip(x, speedups.iterrows()):
        ax.text(xi, max(r["deepcopy_s"], r["clone_s"]), f"{r['speedup']:.1f}x",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_yscale("log"); ax.set_ylabel("time, all features (s)")
    ax.set_xticks(x); ax.set_xticklabels(speedups["dataset"])
    ax.set_title("Same algorithm, same results: implementation speedup")
    ax.legend()
    _save(fig, path_no_ext)


def plot_binning_sensitivity(binning: Dict[str, pd.DataFrame], dataset: str, path_no_ext: str) -> None:
    """Binning-strategy sensitivity, saved as separate images so any one can
    stand alone in a paper:
      * ``{path_no_ext}_accuracy.png``          - accuracy vs bins, per strategy;
      * ``{path_no_ext}_stability.png``         - cross-strategy Spearman rho
        (fixed bins) -- every position in the ranking weighted equally;
      * ``{path_no_ext}_stability_rbo.png``     - cross-strategy Rank-Biased
        Overlap (fixed bins) -- top-weighted;
      * ``{path_no_ext}_bins_stability.png``    - cross-bins-count Spearman rho
        (fixed strategy);
      * ``{path_no_ext}_bins_stability_rbo.png``- cross-bins-count Rank-Biased
        Overlap (fixed strategy).
    """
    summary, stability = binning["summary"], binning["stability"]
    bins_stability = binning["bins_stability"]
    strategies = list(summary["strategy"].unique())

    # accuracy vs bins per strategy
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, s in enumerate(strategies):
        d = summary[summary["strategy"] == s].sort_values("bins")
        ax.plot(d["bins"], d["accuracy"], "o-", color=COLOR_SCHEME[i], label=s)
    ax.set_xlabel("bins"); ax.set_ylabel("test accuracy")
    ax.set_title(f"Accuracy vs binning - {PRETTY_NAMES.get(dataset, dataset)}")
    ax.legend(title="strategy")
    fig.tight_layout()
    _save(fig, f"{path_no_ext}_accuracy")

    # cross-strategy ranking agreement vs bins -- one image per metric
    pair_label = stability["strategy_a"] + "–" + stability["strategy_b"]
    for metric, ylabel, fname in [("spearman", r"Spearman $\rho$ between strategies", "stability"),
                                  ("rbo", "RBO between strategies", "stability_rbo")]:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for i, pl in enumerate(pair_label.unique()):
            d = stability[pair_label == pl].sort_values("bins")
            ax.plot(d["bins"], d[metric], "o-", color=COLOR_SCHEME[i], label=pl)
        ax.set_xlabel("bins"); ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(stability["bins"].unique()))
        if metric == "rbo":
            ax.set_ylim(0, 1.02)
        ax.set_title(f"Ranking stability across strategies - {PRETTY_NAMES.get(dataset, dataset)}")
        ax.legend()
        fig.tight_layout()
        _save(fig, f"{path_no_ext}_{fname}")

    # cross-bins-count ranking agreement per strategy -- one image per metric
    bpair_label = bins_stability["bins_a"].astype(str) + "–" + bins_stability["bins_b"].astype(str)
    for metric, ylabel, fname in [("spearman", r"Spearman $\rho$ between bin counts", "bins_stability"),
                                  ("rbo", "RBO between bin counts", "bins_stability_rbo")]:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for i, s in enumerate(strategies):
            mask = bins_stability["strategy"] == s
            d = bins_stability[mask].assign(pair=bpair_label[mask]).sort_values(["bins_a", "bins_b"])
            ax.plot(d["pair"], d[metric], "o-", color=COLOR_SCHEME[i], label=s)
        ax.set_xlabel("bin-count pair"); ax.set_ylabel(ylabel)
        if metric == "rbo":
            ax.set_ylim(0, 1.02)
        ax.set_title(f"Ranking stability across bin counts - {PRETTY_NAMES.get(dataset, dataset)}")
        ax.legend(title="strategy")
        fig.tight_layout()
        _save(fig, f"{path_no_ext}_{fname}")


def plot_scores(scores: Dict[str, object], dataset: str, path_no_ext: str) -> None:
    """Reproduce the paper's usefulness-score figure (Fig. 4 / 5): one image per
    bin count (``{path_no_ext}_{bins}bins.png``), horizontal bars at the median
    with Q1–Q3 whiskers and mean accuracy."""
    per_bins = scores["per_bins"]
    bins_list = sorted(per_bins)
    color = {"california": COLOR_SCHEME[5], "adult_income": COLOR_SCHEME[1],
             "bike_sharing": COLOR_SCHEME[3]}.get(dataset, COLOR_SCHEME[0])

    for bins in bins_list:
        d = per_bins[bins]
        fig, ax = plt.subplots(figsize=(5, 6))
        y = np.arange(len(d["labels"]))
        ax.barh(y, d["median"], color=color, alpha=0.4, edgecolor="black", height=0.6)
        for yi, q1, q3 in zip(y, d["q1"], d["q3"]):          # Q1–Q3 whisker with end ticks
            ax.plot([q1, q3], [yi, yi], color="black", lw=1.5)
            ax.plot([q1, q1], [yi - 0.12, yi + 0.12], color="black", lw=1)
            ax.plot([q3, q3], [yi - 0.12, yi + 0.12], color="black", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(d["labels"])
        ax.set_xlabel("Score"); ax.set_title(f"{bins} bins")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.text(0.5, -0.12, f"Avg Acc = {d['mean_accuracy']:.3f}", transform=ax.transAxes,
                ha="center", va="top", bbox=dict(facecolor="white", edgecolor="black"))
        fig.suptitle(f"Usefulness score - {PRETTY_NAMES.get(dataset, dataset)}", fontweight="bold", y=1.04)
        fig.tight_layout()
        _save(fig, f"{path_no_ext}_{bins}bins")
