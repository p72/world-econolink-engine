"""個別方程式の推定 (論文 第5章・第6章): 説明変数のみ標準化 → statsmodels OLS → pickle 保存。"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
# statsmodels.api は tsa 系の拡張DLLまで読み込むため、環境によっては (アプリ制御ポリシー等で) 失敗する。
# OLS に必要なモジュールだけを直接 import する。
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant

from .equations import EQUATIONS, Equation
from .variables import add_derived, col


class StandardScaler:
    """sklearn.preprocessing.StandardScaler と同じ属性名 (mean_, scale_) を持つ最小実装。"""

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


def frame_getter(df: pd.DataFrame):
    def g(var: str, lag: int):
        return df[col(var)].shift(lag)
    return g


def build_xy(eq: Equation, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    g = frame_getter(df)
    X = pd.DataFrame({name: f(g) for name, f in eq.features.items()}, index=df.index)
    y = eq.target_series(g).rename(eq.target)
    data = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return data[list(eq.features)], data[eq.target]


@dataclass
class FitReport:
    key: str
    n_train: int
    r2: float
    adj_r2: float
    dw: float
    rmse_test: float | None
    status: str = "ok"


def fit_equation(eq: Equation, df: pd.DataFrame, train_end: str, train_start: str | None = None,
                 min_obs: int = 36):
    X, y = build_xy(eq, df)
    if train_start:
        X, y = X.loc[train_start:], y.loc[train_start:]
    Xtr, ytr = X.loc[:train_end], y.loc[:train_end]
    if len(ytr) < min_obs:
        return None, FitReport(eq.key, len(ytr), np.nan, np.nan, np.nan, None,
                               status=f"skip (有効サンプル {len(ytr)} < {min_obs})")
    scaler = StandardScaler().fit(Xtr.values)
    Ztr = pd.DataFrame(scaler.transform(Xtr.values), index=Xtr.index, columns=Xtr.columns)
    model = OLS(ytr, add_constant(Ztr, has_constant="add")).fit()

    Xte, yte = X.loc[train_end:].iloc[1:], y.loc[train_end:].iloc[1:]
    rmse = None
    if len(yte):
        Zte = add_constant(pd.DataFrame(scaler.transform(Xte.values), index=Xte.index,
                                           columns=Xte.columns), has_constant="add")
        rmse = float(np.sqrt(np.mean((yte - model.predict(Zte)) ** 2)))
    report = FitReport(eq.key, int(model.nobs), model.rsquared, model.rsquared_adj,
                       float(durbin_watson(model.resid)), rmse)
    return {"model": model, "scaler": scaler, "features": list(eq.features),
            "target": eq.target, "mode": eq.mode}, report


def train_all(df: pd.DataFrame, models_dir: Path, train_end: str = "2025-01-01",
              train_start: str | None = "2007-01-01", verbose: bool = True) -> list[FitReport]:
    df = add_derived(df)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for eq in EQUATIONS.values():
        path = models_dir / f"{eq.key}.pkl"
        try:
            bundle, rep = fit_equation(eq, df, train_end, train_start)
        except KeyError as e:
            bundle, rep = None, FitReport(eq.key, 0, np.nan, np.nan, np.nan, None, f"skip (列なし: {e.args[0]})")
        if bundle is not None:
            with open(path, "wb") as fh:
                pickle.dump(bundle, fh)
        else:
            path.unlink(missing_ok=True)  # 別データで学習した古いモデルを残さない
        reports.append(rep)
    if verbose:
        print_reports(reports)
    return reports


def print_reports(reports: list[FitReport]) -> None:
    print(f"{'model':34s} {'N':>4s} {'R2':>6s} {'adjR2':>6s} {'DW':>5s} {'testRMSE':>9s}  status")
    for r in reports:
        rm = "" if r.rmse_test is None else f"{r.rmse_test:9.3f}"
        print(f"{r.key:34s} {r.n_train:4d} {r.r2:6.3f} {r.adj_r2:6.3f} {r.dw:5.2f} {rm:>9s}  {r.status}")


def load_models(models_dir: Path) -> dict:
    out = {}
    for key in EQUATIONS:
        p = models_dir / f"{key}.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                out[key] = pickle.load(fh)
    return out
