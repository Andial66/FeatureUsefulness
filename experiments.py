from __future__ import annotations

import math
import time
import random
import warnings
import tracemalloc
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
from sklearn.neural_network import MLPClassifier

# scipy ships with scikit-learn, so these are always available.
from scipy.stats import spearmanr, kendalltau, wilcoxon


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
    "integrated_gradients": COLOR_SCHEME[4],
    "mdi":         COLOR_SCHEME[5],
}


@dataclass
class Config:
    """All knobs of the experiments in one reproducible place."""

    seed: int = 42                       # master seed
    n_models: int = 20                   # trees trained per configuration (paper uses 20)
    bins_grid: Tuple[int, ...] = (3, 4, 5, 6)
    strategies: Tuple[str, ...] = ("uniform", "quantile", "kmeans")
    test_size: float = 0.20
    # per-dataset leaf regularisation, matching the paper (100*bins / 150*bins).
    leaves_per_bin: Dict[str, int] = field(
        default_factory=lambda: {"california": 100, "bike_sharing": 100, "adult_income": 150}
    )
    # How many instances to use for the (sampling-based) importance methods.
    importance_sample: int = 200
    results_dir: str = "results_ext"


def set_all_seeds(seed: int) -> None:
    """Seed every RNG we rely on, for run-to-run reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def set_plot_style() -> None:
    """A clean, paper-friendly matplotlib style (colour-blind safe, legible)."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
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
        threshold = dec_tree.tree_.threshold[node]              # FIX-1: use dec_tree, not global clf
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
    tree_all_true.condition(feature, lo)                        # FIX-3: start at true domain min
    for val in range(lo + 1, hi + 1):
        nxt = clone(tree)
        nxt.condition(feature, val)
        nxt.conjunction(tree_all_true)
        tree_all_true = nxt
    acum = tree_all_true.count()

    # Same, but for the negation: entities that are always False over `feature`.
    tree_all_false = clone(tree)
    tree_all_false.negate()
    tree_all_false.condition(feature, lo)                       # FIX-3
    for val in range(lo + 1, hi + 1):
        nxt = clone(tree)
        nxt.condition(feature, val)
        nxt.negate()
        nxt.conjunction(tree_all_false)
        tree_all_false = nxt
    acum += tree_all_false.count()

    # Total size of the (categorical) entity space.
    total = 1
    for f in domains:                                          # FIX-2: iterate domains, not global
        a, b = domains[f]
        total *= (math.floor(b) - math.ceil(a) + 1)

    return total - acum


def compute_score(tree: Tree, feature: str, domains: Dict[str, Tuple[float, float]]) -> int:
    """Usefulness score of ``feature``

    Counts the number of entities ``e`` for which ``feature`` is *useful*, i.e.
    for which some value change ``e -> e[feature=b]`` flips the prediction.  It
    equals ``|entity space| - #{e : prediction is constant over feature}``.

    Uses the fast copy-free scorer (``Tree.clone``); the result is identical to
    the ``copy.deepcopy`` prototype but ~5x faster (that overhead was purely an
    implementation artifact, not the algorithm — see ``profile_scorer_speedup``).
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

    # Encode *all* columns as ordinal integers (as the notebook does).
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
    """Train one decision tree for ``dataset`` with the requested discretisation.

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
            domains["season"] = (1, 4)          # FIX: real values are 1..4, not (0, 1)
            domains["weathersit"] = (1, 4)      # FIX: real values are 1..4, not (0, 4)
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
#   * usefulness            - our logic-based score (exact, on the tree itself);
#   * mdi                   - Gini / Mean Decrease in Impurity (free, tree-native);
#   * permutation           - model-agnostic, measured on held-out data;
#   * shap                  - TreeSHAP, the baseline already used in the paper;
#   * lime                  - local surrogates, aggregated to a global score;
#   * integrated_gradients  - needs a *differentiable* model, so it is computed on
#                             an MLP surrogate (a discussion point in itself: the
#                             usefulness score needs no surrogate).


def importance_mdi(bundle: ModelBundle) -> np.ndarray:
    """Mean Decrease in Impurity (a.k.a. Gini importance), native to the tree."""
    return np.asarray(bundle.clf.feature_importances_, dtype=float)


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
    import shap  # lazy, guarded by the caller
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
    continuous discretiser is switched off.
    """
    from lime.lime_tabular import LimeTabularExplainer  # lazy, guarded by caller
    X = bundle.X_train.to_numpy(dtype=float)
    d = X.shape[1]
    explainer = LimeTabularExplainer(
        X, mode="classification", feature_names=list(bundle.features),
        categorical_features=list(range(d)), discretize_continuous=False,
        random_state=seed)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)

    agg = np.zeros(d)
    label = int(bundle.clf.classes_[-1])              # explain the positive class
    for i in idx:
        exp = explainer.explain_instance(X[i], bundle.clf.predict_proba,
                                         num_features=d, labels=(label,))
        for fidx, w in exp.as_map()[label]:
            agg[fidx] += abs(w)
    return agg / len(idx)


def importance_integrated_gradients(bundle: ModelBundle, sample: int = 100, steps: int = 32,
                                    seed: int = 42, hidden=(64,), baseline: str = "median"
                                    ) -> Tuple[np.ndarray, float]:
    """Integrated Gradients on a differentiable MLP *surrogate* of the tree.

    Integrated Gradients requires a differentiable model, which a decision tree
    is not.  We therefore train a small MLP surrogate on the same binned data and
    integrate its gradients along the straight-line path from a baseline to each
    input, using the midpoint (Riemann) rule and central finite differences.
    (Swap in captum + a torch model here if a GPU/torch backend is available.)

    Returns ``(importance_vector, surrogate_test_accuracy)``.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    surrogate = make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=hidden, max_iter=400,
                                            random_state=seed))
    surrogate.fit(bundle.X_train, bundle.y_train)
    surr_acc = float((surrogate.predict(bundle.X_test) == bundle.y_test).mean())

    Xtr = bundle.X_train.to_numpy(dtype=float)
    d = Xtr.shape[1]
    base = np.median(Xtr, axis=0) if baseline == "median" else np.zeros(d)
    pos = list(surrogate.classes_).index(surrogate.classes_[-1])
    F = lambda M: surrogate.predict_proba(M)[:, pos]      # scalar output to attribute

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=min(sample, len(Xtr)), replace=False)
    alphas = (np.arange(1, steps + 1) - 0.5) / steps      # midpoint rule
    h = 1e-2

    ig = np.zeros(d)
    for i in idx:
        x = Xtr[i]
        diff = x - base
        path = base[None, :] + alphas[:, None] * diff[None, :]   # (steps, d)
        grad = np.zeros((steps, d))
        for j in range(d):
            Xp = path.copy(); Xp[:, j] += h
            Xm = path.copy(); Xm[:, j] -= h
            grad[:, j] = (F(Xp) - F(Xm)) / (2 * h)               # central difference
        ig += np.abs(diff * grad.mean(axis=0))                   # |IG_i| for this row
    return ig / len(idx), surr_acc


# Registry of every importance method (name -> callable(bundle, cfg)).
def compute_all_importances(bundle: ModelBundle, cfg: Config,
                            methods: Sequence[str] = ("usefulness", "mdi", "permutation",
                                                      "shap", "lime", "integrated_gradients"),
                            ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    """Compute the requested importance vectors, timing each one.

    Returns ``(importances, timings_seconds, extras)``.  Methods whose optional
    dependency is missing are skipped with a warning rather than crashing.
    """
    importances: Dict[str, np.ndarray] = {}
    timings: Dict[str, float] = {}
    extras: Dict[str, float] = {}

    for m in methods:
        t0 = time.perf_counter()
        try:
            if m == "usefulness":
                importances[m] = usefulness_scores(bundle.clf, bundle.domains, bundle.features)
            elif m == "mdi":
                importances[m] = importance_mdi(bundle)
            elif m == "permutation":
                importances[m] = importance_permutation(bundle, seed=cfg.seed)
            elif m == "shap":
                importances[m] = importance_shap(bundle, sample=cfg.importance_sample, seed=cfg.seed)
            elif m == "lime":
                importances[m] = importance_lime(bundle, sample=min(cfg.importance_sample, 100), seed=cfg.seed)
            elif m == "integrated_gradients":
                ig, acc = importance_integrated_gradients(bundle, sample=min(cfg.importance_sample, 100), seed=cfg.seed)
                importances[m] = ig
                extras["ig_surrogate_acc"] = acc
            else:
                raise ValueError(f"Unknown importance method {m!r}")
            timings[m] = time.perf_counter() - t0
        except Exception as exc:                      # optional dep missing / method failed
            warnings.warn(f"Importance method {m!r} skipped: {exc!r}")
    return importances, timings, extras


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
# 4.  Comparing rankings: correlations, top-k overlap, ground-truth agreement
# ---------------------------------------------------------------------------
#
# Because the methods live on wildly different scales, we compare the *rankings*
# they induce, not the raw numbers.  Three complementary views:
#   * rank correlation (Spearman / Kendall) between every pair of methods;
#   * top-k overlap of each method with the usefulness score -- this is exactly
#     the metric of Table 1 in the paper, here extended to every method;
#   * agreement with the domain "ground-truth" ranking stated in the paper.


# Ground-truth importance tiers taken verbatim from Section 4 of the paper
# (higher value = more important).  Only the features the paper explicitly
# mentions are listed; the rest are left out of the correlation.  Keys use the
# "logical" feature name; :func:`_resolve` maps them onto the actual (possibly
# "_binned") columns of a bundle.
GROUND_TRUTH: Dict[str, Dict[str, int]] = {
    "california": {"MedInc": 4, "Longitude": 3, "Latitude": 3,
                   "HouseAge": 2, "Population": 1, "AveBedrms": 1},
    "bike_sharing": {"hr": 4, "temp": 3, "hum": 3, "weekday": 1, "holiday": 1},
    "adult_income": {"education-num": 3, "capital-gain": 3, "relationship": 3,
                     "fnlwgt": 1, "race": 1, "education": 1},
}


def _resolve(name: str, features: Sequence[str]) -> Optional[str]:
    """Map a logical ground-truth name to the actual column in ``features``."""
    if name in features:
        return name
    if name + "_binned" in features:
        return name + "_binned"
    return None


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


def ground_truth_agreement(importances: Dict[str, np.ndarray], dataset: str,
                           features: Sequence[str]) -> pd.DataFrame:
    """How well each method matches the paper's stated ground-truth ranking.

    Reports, per method:
      * ``spearman_gt`` - Spearman correlation with the ground-truth tiers,
        computed over the annotated features only;
      * ``top1_hit``    - 1 if the ground-truth #1 feature is the method's #1;
      * ``top3_recall`` - fraction of the top ground-truth tier found in the
        method's global top-3.
    """
    gt = GROUND_TRUTH[dataset]
    # Resolve names and keep only those present in this model.
    resolved = {(_resolve(k, features)): v for k, v in gt.items() if _resolve(k, features)}
    gt_feats = list(resolved)
    gt_scores = np.array([resolved[f] for f in gt_feats], dtype=float)
    top_gt_value = max(resolved.values())
    top_gt_feats = {f for f, v in resolved.items() if v == top_gt_value}

    rows = {}
    for name, imp in importances.items():
        imp = np.asarray(imp, dtype=float)
        sub = np.array([imp[list(features).index(f)] for f in gt_feats])
        rho = spearmanr(sub, gt_scores).correlation if len(gt_feats) > 2 else np.nan
        rank = ranking_from_importance(imp, features)
        top1_hit = int(rank[0] in top_gt_feats)
        top3_recall = len(set(rank[:3]) & top_gt_feats) / len(top_gt_feats)
        rows[name] = {"spearman_gt": rho, "top1_hit": top1_hit, "top3_recall": top3_recall}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 5.  Runtime and memory analysis
# ---------------------------------------------------------------------------
#
# We measure wall-clock time with ``time.perf_counter`` (best-of-``repeat`` to
# damp noise) and *peak* Python allocation with ``tracemalloc`` -- both from the
# standard library, so the measurement itself adds no dependency and is fully
# reproducible.  The two natural "size" axes for the usefulness algorithm are the
# number of tree nodes and the per-feature domain size (number of bins), so those
# are what we sweep and plot against.


def entity_space_log10(domains: Dict[str, Tuple[float, float]]) -> float:
    """log10 of the categorical entity-space size (it easily overflows float)."""
    return float(sum(math.log10(math.floor(b) - math.ceil(a) + 1) for a, b in domains.values()))


def profile_usefulness(bundle: ModelBundle, repeat: int = 5) -> Dict[str, float]:
    """Time and peak-memory of computing *all* usefulness scores for one model."""
    tree = dec_tree_to_my_tree(bundle.clf, bundle.domains, bundle.features)

    # --- timing: best of `repeat` runs ---
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for f in bundle.features:
            compute_score(tree, f, bundle.domains)
        best = min(best, time.perf_counter() - t0)

    # --- peak memory of one full pass ---
    tracemalloc.start()
    tracemalloc.reset_peak()
    for f in bundle.features:
        compute_score(tree, f, bundle.domains)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "dataset": bundle.dataset, "bins": bundle.bins, "strategy": bundle.strategy,
        "n_features": len(bundle.features), "n_nodes": int(bundle.clf.tree_.node_count),
        "max_depth": int(bundle.clf.get_depth()), "accuracy": bundle.accuracy,
        "entity_space_log10": entity_space_log10(bundle.domains),
        "time_total_s": best, "time_per_feature_s": best / len(bundle.features),
        "peak_mem_mb": peak / 1024 / 1024,
    }


def scaling_experiment(dataset: str, cfg: Config, strategy: str = "uniform",
                       bins_list: Optional[Sequence[int]] = None,
                       leaves_list: Sequence[int] = (50, 100, 200, 400, 800, 1600, 3200, 6400),
                       seed: Optional[int] = None) -> pd.DataFrame:
    """Sweep #bins and tree size (max_leaf_nodes) and profile the usefulness algo.

    Two sweeps are combined into one tidy table:
      * vary ``bins`` at a fixed, generous leaf budget (cost vs domain size);
      * vary ``max_leaf_nodes`` at fixed bins (cost vs tree size).
    """
    seed = cfg.seed if seed is None else seed
    bins_list = list(cfg.bins_grid) if bins_list is None else list(bins_list)
    rows = []

    # Sweep 1: number of bins (tree left large so it can actually use them).
    for bins in bins_list:
        b = train_tree(dataset, bins=bins, strategy=strategy, seed=seed, leaves=100 * bins)
        r = profile_usefulness(b)
        r["sweep"] = "bins"
        rows.append(r)

    # Sweep 2: tree size, bins fixed at the middle of the grid.
    fixed_bins = bins_list[len(bins_list) // 2]
    for leaves in leaves_list:
        b = train_tree(dataset, bins=fixed_bins, strategy=strategy, seed=seed, leaves=leaves)
        r = profile_usefulness(b)
        r["sweep"] = "leaves"
        rows.append(r)

    return pd.DataFrame(rows)


def method_runtime_comparison(bundle: ModelBundle, cfg: Config,
                              methods: Sequence[str] = ("usefulness", "mdi", "permutation",
                                                        "shap", "lime", "integrated_gradients"),
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
            imp_one, timings, _ = compute_all_importances(bundle, cfg, methods=(m,))
            if m not in timings:                       # method skipped (missing dep)
                ok = False
                break
            best = min(best, timings[m])
        if ok:
            rows.append({"method": m, "time_s": best,
                         "n_nodes": int(bundle.clf.tree_.node_count),
                         "n_features": len(bundle.features)})
    return pd.DataFrame(rows)


# --- (a) implementation speedup: copy-free scorer vs copy.deepcopy ----------
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


# --- (b) runtime as a function of the number of instances processed ---------
def runtime_vs_dataset_size(bundle: ModelBundle, cfg: Config,
                            sizes: Sequence[int] = (25, 50, 100, 200, 400, 800),
                            methods: Sequence[str] = ("usefulness", "mdi", "permutation", "shap", "lime"),
                            repeat: int = 1) -> pd.DataFrame:
    """Wall-clock of each method vs the number of instances it must process.

    The usefulness score reads only the tree + domains, so it is **data-free**:
    its runtime is flat in the dataset size, while SHAP/permutation/LIME/IG grow.
    """
    rng = np.random.RandomState(cfg.seed)

    def _run(m: str, n: int) -> None:
        if m == "usefulness":
            usefulness_scores(bundle.clf, bundle.domains, bundle.features)
        elif m == "mdi":
            importance_mdi(bundle)
        elif m == "permutation":
            k = min(n, len(bundle.X_test))
            idx = rng.choice(len(bundle.X_test), k, replace=False)
            permutation_importance(bundle.clf, bundle.X_test.iloc[idx], bundle.y_test[idx],
                                   n_repeats=5, random_state=cfg.seed, scoring="accuracy")
        elif m == "shap":
            importance_shap(bundle, sample=n, seed=cfg.seed)
        elif m == "lime":
            importance_lime(bundle, sample=n, seed=cfg.seed)
        elif m == "integrated_gradients":
            importance_integrated_gradients(bundle, sample=n, seed=cfg.seed)

    # Warm up each method once (untimed): some backends, e.g. SHAP's numba, pay a
    # large one-time JIT cost on the first call that would otherwise pollute the
    # smallest data point.
    for m in methods:
        try:
            _run(m, min(sizes))
        except Exception:
            pass

    rows = []
    for n in sizes:
        for m in methods:
            best, ok = math.inf, True
            for _ in range(repeat):
                t0 = time.perf_counter()
                try:
                    _run(m, n)
                except Exception as exc:
                    warnings.warn(f"{m} skipped at n={n}: {exc!r}"); ok = False; break
                best = min(best, time.perf_counter() - t0)
            if ok:
                rows.append({"method": m, "n_instances": n, "time_s": best})
    return pd.DataFrame(rows)


# --- (c) time to a stable ranking for the approximate methods ---------------
# Per-method "budget" knob: sample size for the instance-based methods, number
# of repeats for permutation.  usefulness/mdi are exact -> a single point.
_BUDGET_GRID = {
    "shap": (10, 25, 50, 100, 200, 400),
    "lime": (10, 25, 50, 100, 200, 400),
    "integrated_gradients": (10, 25, 50, 100, 200),
    "permutation": (1, 2, 5, 10, 20, 50),
}


def _importance_with_budget(bundle: ModelBundle, method: str, budget: int, cfg: Config) -> np.ndarray:
    if method == "shap":
        return importance_shap(bundle, sample=budget, seed=cfg.seed)
    if method == "lime":
        return importance_lime(bundle, sample=budget, seed=cfg.seed)
    if method == "integrated_gradients":
        return importance_integrated_gradients(bundle, sample=budget, seed=cfg.seed)[0]
    if method == "permutation":
        return importance_permutation(bundle, n_repeats=budget, seed=cfg.seed)
    raise ValueError(method)


def time_to_stable_ranking(bundle: ModelBundle, cfg: Config,
                           methods: Sequence[str] = ("shap", "permutation", "lime"),
                           exact: Sequence[str] = ("usefulness", "mdi")) -> pd.DataFrame:
    """How much wall-clock each approximate method needs before its ranking stops
    changing, versus the exact methods (which are stable by construction).

    For every method we sweep its budget, record runtime and the rank correlation
    of that budget's ranking with the method's own largest-budget ("converged")
    ranking.  Exact methods are added as single points at correlation 1.0.
    """
    rows = []
    for m in methods:
        grid = _BUDGET_GRID.get(m)
        if grid is None:
            continue
        try:                                            # untimed warm-up (e.g. SHAP numba JIT)
            _importance_with_budget(bundle, m, grid[0], cfg)
        except Exception:
            pass
        imps, times = {}, {}
        for budget in grid:
            t0 = time.perf_counter()
            try:
                imps[budget] = _importance_with_budget(bundle, m, budget, cfg)
            except Exception as exc:
                warnings.warn(f"{m} skipped at budget={budget}: {exc!r}")
                continue
            times[budget] = time.perf_counter() - t0
        if not imps:
            continue
        converged = imps[max(imps)]                     # ranking at the largest budget
        for budget in sorted(imps):
            rho = spearmanr(imps[budget], converged).correlation
            rows.append({"method": m, "budget": budget, "time_s": times[budget],
                         "spearman_to_converged": 1.0 if np.isnan(rho) else rho})

    # exact methods: a single converged point at their measured cost
    for m in exact:
        t0 = time.perf_counter()
        try:
            usefulness_scores(bundle.clf, bundle.domains, bundle.features) if m == "usefulness" else importance_mdi(bundle)
        except Exception:
            continue
        rows.append({"method": m, "budget": np.nan, "time_s": time.perf_counter() - t0,
                     "spearman_to_converged": 1.0})
    return pd.DataFrame(rows)


# --- (d) quality vs cost, for the Pareto view -------------------------------
def quality_cost_table(bundle: ModelBundle, cfg: Config,
                       methods: Sequence[str] = ("usefulness", "mdi", "permutation",
                                                 "shap", "lime", "integrated_gradients")
                       ) -> pd.DataFrame:
    """One row per method: its runtime and its agreement with the ground truth,
    on a single model — the two axes of the quality/cost Pareto plot."""
    imp, _, _ = compute_all_importances(bundle, cfg, methods=methods)      # quality
    gt = ground_truth_agreement(imp, bundle.dataset, bundle.features)
    rt = method_runtime_comparison(bundle, cfg, methods=methods, repeat=3)  # warm cost (min-of-3)
    cost = dict(zip(rt["method"], rt["time_s"]))
    return pd.DataFrame([{"method": m, "time_s": cost[m],
                          "spearman_gt": float(gt.loc[m, "spearman_gt"])}
                         for m in imp if m in cost])


# ---------------------------------------------------------------------------
# 6.  Binning-strategy sensitivity analysis
# ---------------------------------------------------------------------------
#
# We re-run the usefulness experiment under ``uniform`` / ``quantile`` / ``kmeans``
# discretisation and ask three questions:
#   * does model accuracy depend on the strategy?
#   * does the *ranking* the usefulness score induces depend on the strategy?
#     (measured by cross-strategy Spearman correlation and top-3 overlap);
#   * does agreement with the domain ground truth depend on the strategy?
# This directly quantifies how much the paper's discretisation choice matters.


def mean_usefulness_importance(dataset: str, bins: int, strategy: str,
                               n_models: int, seed: int, leaves: int) -> Tuple[np.ndarray, List[str], float]:
    """Average (sum-normalised) usefulness importance over ``n_models`` trees.

    Averaging the normalised vectors makes the aggregate scale-invariant, so
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
    """Full binning sensitivity run, returning three tidy tables.

    Returns ``{"summary", "stability", "importance"}`` DataFrames, suitable both
    for the plots in :func:`plot_binning_sensitivity` and for saving to CSV.
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
            gt = ground_truth_agreement({"usefulness": v}, dataset, features).loc["usefulness"]
            summary_rows.append({"dataset": dataset, "strategy": strategy, "bins": bins,
                                 "accuracy": acc, "spearman_gt": gt["spearman_gt"],
                                 "top1_hit": gt["top1_hit"], "top3_recall": gt["top3_recall"]})
            for f, val in zip(features, v):
                importance_rows.append({"strategy": strategy, "bins": bins,
                                        "feature": f.replace("_binned", ""), "importance": val})

    # cross-strategy stability at each bin count
    stability_rows = []
    for bins in bins_grid:
        for i, sa in enumerate(strategies):
            for sb in strategies[i + 1:]:
                va, vb = mean_imp[(sa, bins)], mean_imp[(sb, bins)]
                rho = spearmanr(va, vb).correlation
                ra = ranking_from_importance(va, features_ref)
                rb = ranking_from_importance(vb, features_ref)
                overlap3 = len(set(ra[:3]) & set(rb[:3]))
                stability_rows.append({"dataset": dataset, "bins": bins,
                                       "strategy_a": sa, "strategy_b": sb,
                                       "spearman": rho, "top3_overlap": overlap3})

    return {"summary": pd.DataFrame(summary_rows),
            "stability": pd.DataFrame(stability_rows),
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
                              methods: Sequence[str] = ("usefulness", "mdi", "permutation",
                                                        "shap", "lime", "integrated_gradients"),
                              n_models: Optional[int] = None, seed: Optional[int] = None
                              ) -> Dict[str, object]:
    """Compare the usefulness ranking with every method over ``n_models`` trees.

    Aggregates (like the paper's Table 1) across models and returns:
      * ``topk``          - mean top-k overlap of each method with usefulness;
      * ``corr_spearman`` - mean pairwise Spearman rank-correlation matrix;
      * ``gt``            - mean ground-truth agreement per method;
      * ``gt_ci``         - mean + 95% bootstrap CI of the ground-truth Spearman,
                            across the ``n_models`` trees (for error bars);
      * ``wilcoxon``      - paired Wilcoxon signed-rank test of usefulness vs SHAP
                            on the per-model ground-truth agreement;
      * ``mean_importance`` - mean normalised importance per (feature, method);
      * ``features``      - feature order (by mean usefulness), for plotting.
    """
    n_models = cfg.n_models if n_models is None else n_models
    seed = cfg.seed if seed is None else seed
    leaves = cfg.leaves_per_bin[dataset] * bins
    rng = random.Random(seed)

    topk_frames, corr_frames, gt_frames = [], [], []
    gt_per_model: Dict[str, List[float]] = {}          # per-model spearman_gt, for CIs / tests
    imp_accum: Dict[str, np.ndarray] = {}
    features: List[str] = []

    for _ in range(n_models):
        s = rng.randint(0, 100_000)
        b = train_tree(dataset, bins=bins, strategy="uniform", seed=s, leaves=leaves)
        features = b.features
        imp, _, _ = compute_all_importances(b, cfg, methods=methods)
        present = [m for m in methods if m in imp]

        topk_frames.append(topk_intersection_vs(imp, features).loc[present])
        corr_frames.append(rank_correlation_matrix({m: imp[m] for m in present}))
        gt_df = ground_truth_agreement(imp, dataset, features).loc[present]
        gt_frames.append(gt_df)
        for m in present:
            gt_per_model.setdefault(m, []).append(float(gt_df.loc[m, "spearman_gt"]))
            nv = normalize_importance(imp[m])
            imp_accum[m] = nv if m not in imp_accum else imp_accum[m] + nv

    mean_importance = pd.DataFrame(
        {m: imp_accum[m] / n_models for m in imp_accum},
        index=[f.replace("_binned", "") for f in features])

    # 95% bootstrap CI of the ground-truth agreement per method
    gt_ci = pd.DataFrame({m: dict(zip(["mean", "ci_lo", "ci_hi"], _bootstrap_ci(v, seed=seed)))
                          for m, v in gt_per_model.items()}).T
    # paired test: is usefulness's agreement different from SHAP's across models?
    wilcox = {}
    if "usefulness" in gt_per_model and "shap" in gt_per_model:
        wilcox["usefulness_vs_shap"] = _paired_wilcoxon(gt_per_model["usefulness"], gt_per_model["shap"])

    return {
        "dataset": dataset, "bins": bins, "n_models": n_models,
        "topk": _mean_of_frames(topk_frames),
        "corr_spearman": _mean_of_frames(corr_frames),
        "gt": _mean_of_frames(gt_frames),
        "gt_ci": gt_ci, "gt_per_model": gt_per_model, "wilcoxon": wilcox,
        "mean_importance": mean_importance,
        "features": [f.replace("_binned", "") for f in features],
    }


def _bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0,
                  alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap (1-alpha) CI of a small sample (drops NaNs)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.RandomState(seed)
    boot = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def _paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, object]:
    """Paired Wilcoxon signed-rank test, robust to all-equal / degenerate inputs."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 1 or np.allclose(a, b):
        return {"stat": float("nan"), "p": 1.0, "n": int(len(a)),
                "note": "rankings identical across models -> no detectable difference"}
    try:
        res = wilcoxon(a, b)
        return {"stat": float(res.statistic), "p": float(res.pvalue), "n": int(len(a))}
    except Exception as exc:                            # e.g. zero_method edge cases
        return {"stat": float("nan"), "p": float("nan"), "n": int(len(a)), "note": repr(exc)}


# ---------------------------------------------------------------------------
# 7b.  Plotting  (all figures saved as PNG)
# ---------------------------------------------------------------------------


def _save(fig, path_no_ext: str) -> None:
    fig.savefig(path_no_ext + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_method_importance_heatmap(comparison: Dict[str, object], path_no_ext: str) -> None:
    """Compact overview: normalised importance, features (sorted) x methods."""
    df = comparison["mean_importance"].copy()
    df = df.reindex(df["usefulness"].sort_values(ascending=False).index)  # sort by usefulness
    # column-normalise so each method's colour spans [0, 1] (relative within method)
    disp = df / df.max(axis=0)
    fig, ax = plt.subplots(figsize=(1.4 * df.shape[1] + 2, 0.42 * df.shape[0] + 2))
    im = ax.imshow(disp.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title(f"Normalised feature importance — {comparison['dataset']} ({comparison['bins']} bins)")
    for i in range(df.shape[0]):                       # annotate cells (text stays ink-coloured)
        for j in range(df.shape[1]):
            ax.text(j, i, f"{disp.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="#111" if disp.values[i, j] < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="importance (col-normalised)")
    _save(fig, path_no_ext)


def plot_rank_correlation_heatmap(comparison: Dict[str, object], path_no_ext: str) -> None:
    """Mean pairwise Spearman correlation between methods (diverging, centred 0)."""
    import matplotlib.colors as mcolors
    M = comparison["corr_spearman"]
    fig, ax = plt.subplots(figsize=(0.9 * len(M) + 2, 0.9 * len(M) + 1.5))
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)          # neutral grey at 0
    im = ax.imshow(M.values, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index)
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#111" if abs(M.values[i, j]) < 0.6 else "white")
    ax.set_title(f"Rank correlation between methods — {comparison['dataset']}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
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
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("features shared with usefulness top-k")
    ax.set_title(f"Overlap with the usefulness ranking — {comparison['dataset']}")
    ax.legend(title="", ncol=len(ks))
    _save(fig, path_no_ext)


def plot_ground_truth_agreement(comparison: Dict[str, object], path_no_ext: str) -> None:
    """Spearman correlation of each method with the paper's ground-truth ranking.

    If the comparison carries a bootstrap CI (``gt_ci``), 95% error bars are drawn.
    """
    gt = comparison["gt"].sort_values("spearman_gt", ascending=False)
    methods = list(gt.index)
    ci = comparison.get("gt_ci")
    fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2, 4.5))
    colors = [METHOD_COLORS.get(m, COLOR_SCHEME[7]) for m in methods]
    heights = gt["spearman_gt"].values
    yerr = None
    if ci is not None:                                 # asymmetric 95% bootstrap CI
        lo = [heights[i] - ci.loc[m, "ci_lo"] for i, m in enumerate(methods)]
        hi = [ci.loc[m, "ci_hi"] - heights[i] for i, m in enumerate(methods)]
        yerr = np.clip(np.array([lo, hi]), 0, None)
    ax.bar(methods, heights, yerr=yerr, capsize=4, color=colors, edgecolor="black", linewidth=0.6,
           error_kw={"ecolor": "#333", "elinewidth": 1})
    top = heights + (yerr[1] if yerr is not None else 0)
    for i, m in enumerate(methods):                    # annotate top-1 hit rate above bars
        ax.text(i, top[i], f"top1={gt['top1_hit'].values[i]:.0%}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("Spearman ρ vs ground truth")
    ax.set_title(f"Agreement with domain ground truth — {comparison['dataset']}")
    ax.set_xticklabels(methods, rotation=20, ha="right")
    _save(fig, path_no_ext)


def plot_scaling(scaling_df: pd.DataFrame, dataset: str, path_no_ext: str) -> None:
    """Runtime and peak memory vs #bins and vs #tree-nodes.

    A 2x2 grid with a single measure per axis (no dual-axis charts): the top row
    is wall-clock time, the bottom row is peak memory; the left column varies the
    number of bins, the right column varies the tree size.
    """
    bins_df = scaling_df[scaling_df["sweep"] == "bins"].sort_values("bins")
    leaf_df = scaling_df[scaling_df["sweep"] == "leaves"].sort_values("n_nodes")
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # top-left: time vs bins
    ax = axs[0, 0]
    ax.plot(bins_df["bins"], bins_df["time_total_s"], "o-", color=COLOR_SCHEME[0])
    ax.set_yscale("log"); ax.set_xlabel("number of bins")
    ax.set_ylabel("time, all features (s)"); ax.set_title("Time vs #bins")

    # top-right: time vs nodes, with an empirical power-law fit (theory check)
    ax = axs[0, 1]
    ax.plot(leaf_df["n_nodes"], leaf_df["time_total_s"], "o-", color=COLOR_SCHEME[0], label="measured")
    x = leaf_df["n_nodes"].to_numpy(float)
    y = leaf_df["time_total_s"].to_numpy(float)
    if len(x) >= 2 and (x > 0).all() and (y > 0).all():
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)   # log-log fit -> exponent
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, np.exp(intercept) * xs ** slope, "--", color=COLOR_SCHEME[7],
                label=fr"fit $\propto |T|^{{{slope:.2f}}}$")
        ax.legend()
    ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xlabel("number of tree nodes")
    ax.set_ylabel("time, all features (s)"); ax.set_title("Time vs tree size")

    # bottom-left: memory vs bins
    ax = axs[1, 0]
    ax.plot(bins_df["bins"], bins_df["peak_mem_mb"], "s-", color=COLOR_SCHEME[1])
    ax.set_xlabel("number of bins"); ax.set_ylabel("peak memory (MB)"); ax.set_title("Memory vs #bins")

    # bottom-right: memory vs nodes
    ax = axs[1, 1]
    ax.plot(leaf_df["n_nodes"], leaf_df["peak_mem_mb"], "s-", color=COLOR_SCHEME[1])
    ax.set_xscale("log"); ax.set_xlabel("number of tree nodes")
    ax.set_ylabel("peak memory (MB)"); ax.set_title("Memory vs tree size")

    fig.suptitle(f"Usefulness-score scaling — {dataset}", fontweight="bold")
    fig.tight_layout()
    _save(fig, path_no_ext)


# ---------------------------------------------------------------------------
# 7d.  Synthetic controlled-feature experiment
# ---------------------------------------------------------------------------
#
# We generate categorical datasets with a *known* target function, so the true
# per-feature relevance is not folklore but something we can compute exactly with
# our own score: we build the decision tree that represents the target and run
# `compute_score` on it.  Three groups of features are planted:
#   * relevant  - appear in the target function;
#   * redundant - a copy of a relevant feature (correlated in the data), absent
#                 from the target itself;
#   * noise     - independent of the target (true usefulness exactly 0).
#
# Two structures are used.  "or_and" -> y = (x0 AND x1) OR x2, which has a graded
# known order (x2 > x0 = x1 > redundant = noise); "mux" -> y = x0 if x2=0 else x1
# (a multiplexer), used with a skewed selector to show that the *distribution-free*
# usefulness score surfaces a feature governing a rare operating mode that the
# data-weighted mean|SHAP| under-reports.

# Sentinel leaves reused when hand-building the exact target trees.
def _leaf(v):
    return Node("True", 1, None, None) if v else Node("False", 0, None, None)


def _true_target_tree(structure: str, domains: Dict[str, Tuple[float, float]]) -> Tree:
    """The decision tree that represents the synthetic target exactly.

    Only x0/x1/x2 are tested; every other feature (redundant, noise) is absent,
    so its usefulness is exactly 0 by construction.
    """
    if structure == "or_and":                       # y = (x0 AND x1) OR x2
        x1n = Node("x1", 0.5, _leaf(False), _leaf(True))     # x2=0, x0=1: y = x1
        x0n = Node("x0", 0.5, _leaf(False), x1n)             # x2=0: y = x0 AND x1
        root = Node("x2", 0.5, x0n, _leaf(True))             # x2=1: y = 1
    elif structure == "mux":                         # y = x0 if x2=0 else x1
        left = Node("x0", 0.5, _leaf(False), _leaf(True))    # x2=0: y = x0
        right = Node("x1", 0.5, _leaf(False), _leaf(True))   # x2=1: y = x1
        root = Node("x2", 0.5, left, right)
    else:
        raise ValueError(structure)
    return Tree(root, domains)


def true_usefulness(structure: str, features: Sequence[str],
                    domains: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Exact usefulness of every feature for the *true* target function."""
    tree = _true_target_tree(structure, domains)
    return {f: compute_score(tree, f, domains) for f in features}


def make_synthetic_dataset(n_samples: int = 8000, n_noise: int = 6, structure: str = "or_and",
                           redundant: bool = True, skew: Optional[float] = None, seed: int = 42
                           ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    """Generate a binary categorical dataset with a known target.

    ``skew`` (if given) is the probability that the selector/third feature ``x2``
    equals 1; a small value makes the ``x2=1`` branch rare in the data (used by
    the "mux" rare-mode demonstration). ``redundant`` adds ``x0_copy = x0``.
    """
    rng = np.random.RandomState(seed)
    x0 = rng.randint(0, 2, n_samples)
    x1 = rng.randint(0, 2, n_samples)
    x2 = (rng.rand(n_samples) < skew).astype(int) if skew is not None else rng.randint(0, 2, n_samples)

    if structure == "or_and":
        y = ((x0 & x1) | x2).astype(int)
    elif structure == "mux":
        y = np.where(x2 == 1, x1, x0).astype(int)
    else:
        raise ValueError(structure)

    cols = {"x0": x0, "x1": x1, "x2": x2}
    redundant_feats = []
    if redundant:
        cols["x0_copy"] = x0.copy()                  # exact copy -> correlated, but not in the target
        redundant_feats = ["x0_copy"]
    noise_feats = [f"noise{i}" for i in range(n_noise)]
    for f in noise_feats:
        cols[f] = rng.randint(0, 2, n_samples)

    X = pd.DataFrame(cols)
    info = {"relevant": ["x0", "x1", "x2"], "redundant": redundant_feats,
            "noise": noise_feats, "structure": structure}
    return X, y.astype(int), info


def synthetic_bundle(X: pd.DataFrame, y: np.ndarray, seed: int = 42,
                     max_leaf_nodes: int = 128, min_samples_leaf: int = 20) -> ModelBundle:
    """Train a tree on a synthetic dataset and wrap it as a :class:`ModelBundle`.

    ``min_samples_leaf`` regularises the tree so it does not split on noise.
    """
    features = list(X.columns)
    domains = {f: (0, 1) for f in features}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    clf = DecisionTreeClassifier(random_state=seed, max_leaf_nodes=max_leaf_nodes,
                                 min_samples_leaf=min_samples_leaf)
    clf.fit(X_train, y_train)
    acc = float((clf.predict(X_test) == y_test).mean())
    return ModelBundle("synthetic", clf, acc, domains, features,
                       X_train.reset_index(drop=True), np.asarray(y_train),
                       X_test.reset_index(drop=True), np.asarray(y_test), 2, "none")


def _useful_fraction(scores: Dict[str, float], domains: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Express scores as the fraction of the input space where the feature is useful."""
    total = 1
    for a, b in domains.values():
        total *= (math.floor(b) - math.ceil(a) + 1)
    return {f: s / total for f, s in scores.items()}


def _synthetic_recovery_case(cfg: Config, seed: int, n_noise: int, redundant: bool,
                             methods: Sequence[str]) -> Dict[str, object]:
    """Train on one 'or_and' dataset and gather per-feature scores + recovery stats."""
    from sklearn.metrics import roc_auc_score
    X, y, info = make_synthetic_dataset(n_samples=8000, n_noise=n_noise,
                                        structure="or_and", redundant=redundant, seed=seed)
    bundle = synthetic_bundle(X, y, seed=seed)
    imp, _, _ = compute_all_importances(bundle, cfg, methods=methods)
    features = bundle.features
    true_u = true_usefulness("or_and", features, bundle.domains)
    emp_u = dict(zip(features, imp["usefulness"]))

    # per-method AUC separating relevant (1) from noise (0); the redundant copy is excluded
    labelled = [f for f in features if f in info["relevant"] or f in info["noise"]]
    labels = np.array([1 if f in info["relevant"] else 0 for f in labelled])
    auc_rows = {}
    for m, vec in imp.items():
        vals = np.array([vec[features.index(f)] for f in labelled])
        auc = roc_auc_score(labels, vals) if len(set(labels)) == 2 else np.nan
        auc_rows[m] = {"auc_relevant_vs_noise": auc,
                       "mean_noise": float(np.mean([vec[features.index(f)] for f in info["noise"]])),
                       "mean_relevant": float(np.mean([vec[features.index(f)] for f in info["relevant"]]))}
    rho = spearmanr([emp_u[f] for f in features], [true_u[f] for f in features]).correlation

    return {"bundle": bundle, "info": info, "importances": imp,
            "true_fraction": _useful_fraction(true_u, bundle.domains),
            "empirical_fraction": _useful_fraction(emp_u, bundle.domains),
            "auc": pd.DataFrame(auc_rows).T, "spearman_true_vs_empirical": float(rho),
            "accuracy": bundle.accuracy}


def run_synthetic_experiment(cfg: Config, seed: int = 42, n_noise: int = 6,
                             methods: Sequence[str] = ("usefulness", "mdi", "permutation",
                                                       "shap", "lime")) -> Dict[str, object]:
    """Controlled recovery on ``y = (x0 AND x1) OR x2``.

    Runs two cases: a *clean* one (no redundant feature) for the headline recovery
    numbers, and one *with* a redundant copy of ``x0`` to show that usefulness is
    a property of the model (the copy absorbs x0's mass when the tree uses it,
    and the total is conserved).
    """
    clean = _synthetic_recovery_case(cfg, seed, n_noise, redundant=False, methods=methods)
    redundant = _synthetic_recovery_case(cfg, seed, n_noise, redundant=True, methods=methods)
    rf, tf = redundant["empirical_fraction"], redundant["true_fraction"]
    redundant["redundancy"] = {"x0": rf["x0"], "x0_copy": rf["x0_copy"],
                               "sum": rf["x0"] + rf["x0_copy"], "true_x0": tf["x0"]}
    return {"clean": clean, "redundant": redundant}


def run_rare_relevance_experiment(cfg: Config, seed: int = 42, n_noise: int = 4,
                                  skew: float = 0.12) -> Dict[str, object]:
    """Mux config with a rare selector: usefulness (distribution-free) keeps the
    rare-mode feature x1 on par with x0, whereas mean|SHAP| (data-weighted) drops
    it toward the noise level.
    """
    X, y, info = make_synthetic_dataset(n_samples=12000, n_noise=n_noise,
                                        structure="mux", redundant=False, skew=skew, seed=seed)
    bundle = synthetic_bundle(X, y, seed=seed)
    imp, _, _ = compute_all_importances(bundle, cfg, methods=("usefulness", "shap"))
    features = bundle.features
    use_n = normalize_importance(imp["usefulness"], how="max")     # max-normalise for comparison
    shap_n = normalize_importance(imp["shap"], how="max")
    rows = []
    for i, f in enumerate(features):
        group = ("relevant" if f in info["relevant"] else
                 "noise" if f in info["noise"] else "redundant")
        rows.append({"feature": f, "group": group,
                     "usefulness": use_n[i], "shap": shap_n[i]})
    return {"bundle": bundle, "info": info, "skew": skew,
            "table": pd.DataFrame(rows), "importances": imp}


def runtime_vs_n_features(cfg: Config, n_features_list: Sequence[int] = (5, 10, 20, 40, 80),
                          n_samples: int = 6000, seed: int = 42, repeat: int = 3) -> pd.DataFrame:
    """Runtime and peak memory of scoring *all* features as the feature count grows.

    Uses the synthetic generator (3 relevant features + the rest noise) so the tree
    stays small: this isolates the dependence on the number of features |X|, which
    is exactly reviewer question 2.f-(i).
    """
    rows = []
    for n in n_features_list:
        X, y, _ = make_synthetic_dataset(n_samples=n_samples, n_noise=max(0, n - 3),
                                         structure="or_and", redundant=False, seed=seed)
        b = synthetic_bundle(X, y, seed=seed)
        tree = dec_tree_to_my_tree(b.clf, b.domains, b.features)
        best = math.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            for f in b.features:
                compute_score(tree, f, b.domains)
            best = min(best, time.perf_counter() - t0)
        tracemalloc.start(); tracemalloc.reset_peak()
        for f in b.features:
            compute_score(tree, f, b.domains)
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rows.append({"n_features": len(b.features), "n_nodes": int(b.clf.tree_.node_count),
                     "time_s": best, "peak_mem_mb": peak / 1024 / 1024})
    return pd.DataFrame(rows)


_SYNTH_GROUP_COLOR = {"relevant": COLOR_SCHEME[0], "redundant": COLOR_SCHEME[1], "noise": "#9aa0a6"}


def _synth_bars(ax, emp, true, info) -> None:
    """Draw empirical-usefulness bars (coloured by planted group) with true markers."""
    group_of = {**{f: "relevant" for f in info["relevant"]},
                **{f: "redundant" for f in info["redundant"]},
                **{f: "noise" for f in info["noise"]}}
    feats = sorted(emp, key=lambda f: -true[f])       # order by true usefulness
    x = np.arange(len(feats))
    ax.bar(x, [emp[f] for f in feats],
           color=[_SYNTH_GROUP_COLOR[group_of[f]] for f in feats], edgecolor="black", linewidth=0.5)
    ax.plot(x, [true[f] for f in feats], "D", color="black", ms=7, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=30, ha="right")


def plot_synthetic_recovery(result: Dict[str, object], path_no_ext: str) -> None:
    """Two panels on ``y = (x0 AND x1) OR x2``: (left) clean recovery — empirical
    usefulness matches the exact/true value and noise sits at 0; (right) with a
    redundant copy of x0, the copy absorbs x0's mass (the model uses it instead),
    while the total relevance is conserved."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    clean, redundant = result["clean"], result["redundant"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [3, 2]})

    _synth_bars(axs[0], clean["empirical_fraction"], clean["true_fraction"], clean["info"])
    axs[0].set_ylabel("fraction of inputs where feature is useful")
    axs[0].set_title("Clean recovery (no redundant feature)")

    _synth_bars(axs[1], redundant["empirical_fraction"], redundant["true_fraction"], redundant["info"])
    r = redundant["redundancy"]
    axs[1].set_title(f"Redundancy: x0 + x0_copy = {r['sum']:.2f} (true x0 = {r['true_x0']:.2f})")

    handles = [Patch(facecolor=_SYNTH_GROUP_COLOR[g], edgecolor="black", label=g)
               for g in ["relevant", "redundant", "noise"]]
    handles.append(Line2D([0], [0], marker="D", color="black", ls="", label="true usefulness"))
    axs[1].legend(handles=handles, fontsize=9)
    fig.suptitle(r"Synthetic controlled-feature experiment: $y = (x_0 \wedge x_1) \vee x_2$",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, path_no_ext)


def plot_synthetic_recovery_auc(auc_df: pd.DataFrame, path_no_ext: str) -> None:
    """Per-method AUC separating the planted relevant features from the noise ones."""
    d = auc_df.sort_values("auc_relevant_vs_noise", ascending=False)
    methods = list(d.index)
    colors = [METHOD_COLORS.get(m, COLOR_SCHEME[7]) for m in methods]
    fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2, 4.5))
    ax.bar(methods, d["auc_relevant_vs_noise"], color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_ylim(0, 1.08); ax.set_ylabel("AUC (relevant vs noise)")
    ax.set_title("Recovery of the planted relevant features")
    for i, v in enumerate(d["auc_relevant_vs_noise"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    _save(fig, path_no_ext)


def plot_rare_relevance(result: Dict[str, object], path_no_ext: str) -> None:
    """Usefulness vs mean|SHAP| for the multiplexer with a rare selector: x1 stays
    high under usefulness (distribution-free) but drops under mean|SHAP|."""
    d = result["table"]
    order = ["x0", "x1", "x2"] + [f for f in d["feature"] if str(f).startswith("noise")]
    d = d.set_index("feature").loc[order].reset_index()
    x = np.arange(len(d)); w = 0.4
    fig, ax = plt.subplots(figsize=(1.1 * len(d) + 2, 4.8))
    ax.bar(x - w / 2, d["usefulness"], width=w, color=COLOR_SCHEME[0], edgecolor="black",
           linewidth=0.5, label="usefulness (distribution-free)")
    ax.bar(x + w / 2, d["shap"], width=w, color=COLOR_SCHEME[1], edgecolor="black",
           linewidth=0.5, label="mean |SHAP| (data-weighted)")
    ax.set_xticks(x); ax.set_xticklabels(d["feature"], rotation=30, ha="right")
    ax.set_ylabel("importance (max-normalised)")
    ax.set_title(f"Rare operating mode: selector $x_2{{=}}1$ in {result['skew']:.0%} of the data\n"
                 r"$y = x_0$ if $x_2{=}0$ else $x_1$")
    ax.legend()
    _save(fig, path_no_ext)


def plot_runtime_vs_n_features(df: pd.DataFrame, path_no_ext: str) -> None:
    """Usefulness runtime vs the number of features, with a fitted power law."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(df["n_features"], df["time_s"], "o-", color=COLOR_SCHEME[0], label="measured")
    x = df["n_features"].to_numpy(float)
    y = df["time_s"].to_numpy(float)
    if len(x) >= 2 and (x > 0).all() and (y > 0).all():
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        ax.plot(x, np.exp(intercept) * x ** slope, "--", color=COLOR_SCHEME[7],
                label=fr"fit $\propto |X|^{{{slope:.2f}}}$")
        ax.legend()
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of features |X|")
    ax.set_ylabel("time to score all features (s)")
    ax.set_title("Usefulness runtime vs number of features\n(3 relevant + noise; tree kept small)")
    _save(fig, path_no_ext)


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
    """Per-method wall-clock cost on one model (log y — costs span decades)."""
    df = runtime_df.sort_values("time_s")
    colors = [METHOD_COLORS.get(m, COLOR_SCHEME[7]) for m in df["method"]]
    fig, ax = plt.subplots(figsize=(1.3 * len(df) + 2, 4.5))
    ax.bar(df["method"], df["time_s"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("time (s, log scale)")
    ax.set_title(f"Runtime per importance method — {dataset}\n"
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


def plot_runtime_vs_dataset_size(df: pd.DataFrame, dataset: str, path_no_ext: str) -> None:
    """Runtime vs #instances processed: usefulness is flat (data-free), the rest grow."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for m in df["method"].unique():
        d = df[df["method"] == m].sort_values("n_instances")
        ax.plot(d["n_instances"], d["time_s"], "o-", color=METHOD_COLORS.get(m, COLOR_SCHEME[7]), label=m)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of instances processed"); ax.set_ylabel("time (s)")
    ax.set_title(f"Runtime vs dataset size — {dataset}\n(usefulness is data-free → flat)")
    ax.legend(ncol=2)
    _save(fig, path_no_ext)


def plot_time_to_stable(df: pd.DataFrame, dataset: str, path_no_ext: str) -> None:
    """Rank agreement with each method's converged ranking vs the wall-clock spent.

    Approximate methods trace a curve (more time → more stable); the exact methods
    are single stars already at agreement 1.0.
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # A log x-axis needs strictly positive times; a method can be timed at ~0 on a
    # fast machine (e.g. MDI), which would otherwise make the axis autoscale explode.
    pos = df["time_s"][df["time_s"] > 0]
    floor = max(float(pos.min()) * 0.5, 1e-6) if len(pos) else 1e-6
    for m in df["method"].unique():
        d = df[df["method"] == m].sort_values("time_s").copy()
        d["time_s"] = d["time_s"].clip(lower=floor)       # keep the log axis well-defined
        color = METHOD_COLORS.get(m, COLOR_SCHEME[7])
        if len(d) == 1:                                   # exact method -> single point
            ax.scatter(d["time_s"], d["spearman_to_converged"], color=color, marker="*",
                       s=260, edgecolor="black", zorder=5, label=f"{m} (exact)")
        else:
            ax.plot(d["time_s"], d["spearman_to_converged"], "o-", color=color, label=m)
    ax.set_xscale("log")
    ax.axhline(0.98, color="grey", ls=":", lw=1)
    # Place the label in axes-fraction x (data y), so it never depends on the
    # data range (using get_xlim() here can blow up the figure on a log axis).
    ax.text(0.01, 0.982, "stable (ρ≥0.98)", transform=ax.get_yaxis_transform(),
            fontsize=8, color="grey", va="bottom", ha="left")
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("rank agreement with converged ranking (ρ)")
    ax.set_title(f"Time to a stable ranking — {dataset}")
    ax.legend(ncol=2, loc="lower right")
    _save(fig, path_no_ext)


def plot_quality_cost_pareto(qc: pd.DataFrame, dataset: str, path_no_ext: str) -> None:
    """Scatter of ground-truth agreement (quality) vs runtime (cost), per method.

    The Pareto-optimal methods (no other method is both faster *and* better) are
    ringed; everything below-and-right of the frontier is dominated.
    """
    d = qc.copy()
    # Pareto frontier: maximise quality, minimise time.
    dominated = set()
    for i, a in d.iterrows():
        for _, b in d.iterrows():
            if (b["time_s"] <= a["time_s"] and b["spearman_gt"] >= a["spearman_gt"]
                    and (b["time_s"] < a["time_s"] or b["spearman_gt"] > a["spearman_gt"])):
                dominated.add(i); break

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Alternate label offsets (points often tie on the y-axis, so labels collide).
    order = list(d.sort_values("time_s").index)
    for rank, i in enumerate(order):
        r = d.loc[i]
        color = METHOD_COLORS.get(r["method"], COLOR_SCHEME[7])
        on_front = i not in dominated
        ax.scatter(r["time_s"], r["spearman_gt"], s=180 if on_front else 110, color=color,
                   edgecolor="black", linewidth=2.0 if on_front else 0.6, zorder=4)
        ax.annotate(r["method"], (r["time_s"], r["spearman_gt"]), textcoords="offset points",
                    xytext=(0, 10 if rank % 2 == 0 else -16), ha="center", fontsize=9)
    # connect the frontier
    front = d.loc[[i for i in d.index if i not in dominated]].sort_values("time_s")
    ax.plot(front["time_s"], front["spearman_gt"], "--", color="grey", zorder=1,
            label="Pareto frontier")
    ax.set_xscale("log")
    ax.margins(x=0.18, y=0.22)                          # headroom so labels don't clip
    ax.set_xlabel("runtime (s, log)  →  cheaper is left")
    ax.set_ylabel("agreement with ground truth (ρ)  →  better is up")
    ax.set_title(f"Quality vs cost — {dataset}")
    ax.legend(loc="lower left")
    _save(fig, path_no_ext)


def plot_binning_sensitivity(binning: Dict[str, pd.DataFrame], dataset: str, path_no_ext: str) -> None:
    """Three panels: accuracy, ground-truth agreement, and cross-strategy stability."""
    summary, stability = binning["summary"], binning["stability"]
    strategies = list(summary["strategy"].unique())
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # (1) accuracy vs bins per strategy
    ax = axs[0]
    for i, s in enumerate(strategies):
        d = summary[summary["strategy"] == s].sort_values("bins")
        ax.plot(d["bins"], d["accuracy"], "o-", color=COLOR_SCHEME[i], label=s)
    ax.set_xlabel("bins"); ax.set_ylabel("test accuracy"); ax.set_title("Accuracy vs binning")
    ax.legend(title="strategy")

    # (2) ground-truth agreement vs bins per strategy
    ax = axs[1]
    for i, s in enumerate(strategies):
        d = summary[summary["strategy"] == s].sort_values("bins")
        ax.plot(d["bins"], d["spearman_gt"], "o-", color=COLOR_SCHEME[i], label=s)
    ax.set_xlabel("bins"); ax.set_ylabel("Spearman ρ vs ground truth")
    ax.set_title("Ranking quality vs binning"); ax.legend(title="strategy")

    # (3) cross-strategy ranking agreement (Spearman) vs bins
    ax = axs[2]
    pair_label = stability["strategy_a"] + "–" + stability["strategy_b"]
    for i, pl in enumerate(pair_label.unique()):
        d = stability[pair_label == pl].sort_values("bins")
        ax.plot(d["bins"], d["spearman"], "o-", color=COLOR_SCHEME[i], label=pl)
    ax.set_xlabel("bins"); ax.set_ylabel("Spearman ρ between strategies")
    ax.set_title("How much the ranking moves with strategy"); ax.legend(title="pair")

    fig.suptitle(f"Binning-strategy sensitivity — {dataset}", fontweight="bold")
    _save(fig, path_no_ext)


def plot_scores(scores: Dict[str, object], dataset: str, path_no_ext: str) -> None:
    """Reproduce the paper's usefulness-score figure (Fig. 4 / 5): one panel per
    bin count, horizontal bars at the median with Q1–Q3 whiskers and mean accuracy."""
    per_bins = scores["per_bins"]
    bins_list = sorted(per_bins)
    titles = {"california": "California Housing", "adult_income": "Adult Income",
              "bike_sharing": "Bike Sharing"}
    color = {"california": COLOR_SCHEME[5], "adult_income": COLOR_SCHEME[1],
             "bike_sharing": COLOR_SCHEME[3]}.get(dataset, COLOR_SCHEME[0])

    fig, axs = plt.subplots(1, len(bins_list), figsize=(5 * len(bins_list), 6))
    if len(bins_list) == 1:
        axs = [axs]
    for ax, bins in zip(axs, bins_list):
        d = per_bins[bins]
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
    fig.suptitle(f"Usefulness score — {titles.get(dataset, dataset)}", fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, path_no_ext)
