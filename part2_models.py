"""Custom estimator used by Part 2 (kept in a module so saved models can be re-loaded)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict


class LateFusion(BaseEstimator, ClassifierMixin):
    """Decision-level fusion.

    blocks: list of (name, column_indices, estimator, param_grid). Each block gets its own model,
    tuned by inner CV on that block's columns only.
    weighted=False -> average the block probabilities (equal weights).
    weighted=True  -> a logistic-regression stacker learns how much to trust each block from
                      out-of-fold block probabilities (speech and text are not assumed equal).
    """

    def __init__(self, blocks, weighted=False, seed=42):
        self.blocks = blocks
        self.weighted = weighted
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        inner = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        self.models_, oofs = [], []
        for name, cols, est, grid in self.blocks:
            gs = GridSearchCV(clone(est), grid, scoring="f1_macro", cv=inner, n_jobs=-1).fit(X[:, cols], y)
            self.models_.append(gs.best_estimator_)
            if self.weighted:
                oofs.append(cross_val_predict(clone(gs.best_estimator_), X[:, cols], y, cv=inner,
                                              method="predict_proba", n_jobs=-1)[:, 1])
        if self.weighted:
            self.stacker_ = LogisticRegression(class_weight="balanced", random_state=self.seed)
            self.stacker_.fit(np.column_stack(oofs), y)
        return self

    def block_probas(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([m.predict_proba(X[:, b[1]])[:, 1] for m, b in zip(self.models_, self.blocks)])

    def predict_proba(self, X):
        P = self.block_probas(X)
        p = self.stacker_.predict_proba(P)[:, 1] if self.weighted else P.mean(axis=1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
