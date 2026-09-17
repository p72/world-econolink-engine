"""シミュレーションエンジン (論文 第7章): ①〜⑳ のブロック再帰的逐次計算。"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .equations import EQUATIONS, STEPS
from .variables import ALL_VARS, add_derived, col, derive

N_BUFFER = 24  # 最長ラグ (前年差のt-1 = t-13 など) を満たす実績バッファ
EXOG_SCENARIO_KEY = {"world.oil_price": "oil_price", "jp.short_rate": "jp_short_rate"}


class CircularReferenceError(RuntimeError):
    """当期値がまだ確定していない変数を当期参照した (= 循環カット漏れ)。"""


@dataclass
class Scenario:
    """外生シナリオ (論文 8.1)。リストが periods より短い場合は最後の値を保持。None はモデル推計。"""
    base_date: str
    periods: int
    jp_short_rate: list[float | None] = field(default_factory=list)
    jp_long_rate: list[float | None] = field(default_factory=list)
    jp_inflation: list[float | None] = field(default_factory=list)
    oil_price: list[float | None] = field(default_factory=list)
    us_short_rate: list[float | None] = field(default_factory=list)
    cpi_mode: str = "model"  # "model" | "scenario"
    # 論文v1には無い拡張。基準期の残差を decay^h で減衰させながら各式に加算 (None = 無効, 論文どおり)
    add_factor_decay: float | None = None

    def value(self, name: str, i: int):
        path = getattr(self, name)
        if not path:
            return None
        return path[min(i, len(path) - 1)]


class Engine:
    def __init__(self, models: dict, data: pd.DataFrame):
        self.models = models
        self.data = add_derived(data).sort_index()
        missing = [k for k in EQUATIONS if k not in models]
        if missing:
            warnings.warn(f"モデル未学習のため前期値を保持する変数: {missing}")

    # ------------------------------------------------------------------ history
    def _initial_history(self, base: pd.Timestamp) -> list[dict]:
        hist = self.data.loc[:base].ffill().iloc[-(N_BUFFER + 1):]
        if hist.index[-1] != base:
            raise ValueError(f"base_date {base.date()} の実績データがありません (最終: {hist.index[-1].date()})")
        return [{v: float(row.get(col(v), np.nan)) for v in ALL_VARS} | {"date": d}
                for d, row in hist.iterrows()]

    # ------------------------------------------------------------------ one period
    def _step(self, history: list[dict], date: pd.Timestamp, i: int, sc: Scenario) -> dict:
        cur: dict = {"date": date}

        def g(var: str, lag: int):
            if lag == 0:
                if var not in cur:
                    raise CircularReferenceError(f"{var} の当期値は未確定です (循環カットが必要)")
                return cur[var]
            return history[-lag][var]

        prev = history[-1]
        for label, items in STEPS:
            for item in items:
                if item.startswith("exog:"):
                    var = item[5:]
                    v = sc.value(EXOG_SCENARIO_KEY[var], i)
                    cur[var] = prev[var] if v is None else float(v)
                elif item == "derive":
                    cur.update(derive(cur))
                else:
                    eq = EQUATIONS[item]
                    value = self._predict(eq, g, prev)
                    if sc.add_factor_decay is not None:
                        value += self._residuals.get(eq.key, 0.0) * sc.add_factor_decay ** (i + 1)
                    cur[eq.target] = value
                    self._apply_overrides(eq.target, cur, sc, i)
        return cur

    def _base_residuals(self, history: list[dict]) -> dict[str, float]:
        """基準期 (period 0) の 実績 − モデル当てはめ値。アドファクターの初期値。"""
        cur, past = history[-1], history[:-1]

        def g(var: str, lag: int):
            return cur[var] if lag == 0 else past[-lag][var]

        res = {}
        for key, eq in EQUATIONS.items():
            if key in self.models:
                r = cur[eq.target] - self._predict(eq, g, past[-1])
                res[key] = r if np.isfinite(r) else 0.0
        return res

    def _predict(self, eq, g, prev) -> float:
        bundle = self.models.get(eq.key)
        if bundle is None:
            return prev[eq.target]
        x = np.array([float(eq.features[name](g)) for name in bundle["features"]])
        z = bundle["scaler"].transform(x)
        beta = np.asarray(bundle["model"].params, dtype=float)
        return float(eq.to_level(beta[0] + beta[1:] @ z, g))

    @staticmethod
    def _apply_overrides(target: str, cur: dict, sc: Scenario, i: int) -> None:
        if target == "jp.long_rate" and (v := sc.value("jp_long_rate", i)) is not None:
            cur[target] = float(v)
        elif target == "jp.inflation" and sc.cpi_mode == "scenario" and (v := sc.value("jp_inflation", i)) is not None:
            cur[target] = float(v)
        elif target == "us.short_rate" and (v := sc.value("us_short_rate", i)) is not None:
            cur[target] = float(v)

    # ------------------------------------------------------------------ run
    def run(self, sc: Scenario, n_actual: int = 12) -> pd.DataFrame:
        base = pd.Timestamp(sc.base_date)
        history = self._initial_history(base)
        self._residuals = self._base_residuals(history) if sc.add_factor_decay is not None else {}
        for i in range(sc.periods):
            date = base + pd.DateOffset(months=i + 1)
            history.append(self._step(history, date, i, sc))

        rows = history[-(sc.periods + n_actual + 1):]
        out = pd.DataFrame(rows).set_index("date")
        out.index.name = "date"
        out.insert(0, "period", np.arange(-n_actual, sc.periods + 1))
        return out.rename(columns={v: col(v) for v in ALL_VARS})
