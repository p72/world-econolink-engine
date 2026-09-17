"""全28方程式の仕様 (論文 付録A 表7・表8) と ①〜⑳ の計算順序 (表5)。

各特徴量は g(var, lag) を受け取る関数として書く。
  - 学習時:       g = DataFrame の列を shift(lag) したもの (pd.Series)
  - シミュレーション時: g = history からラグ値を引く (float)
同じ関数を両方で使うので、学習とシミュレーションで特徴量の定義が食い違うことがない。

論文の表から読み取れない細部 (表の途中で切れている項目等) は、
計算順序と矛盾しない形で本実装が補ったもの。該当箇所には「※補完」とコメントしている。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Getter = Callable[[str, int], object]
Feature = Callable[[Getter], object]


def L(v: str, k: int = 0) -> Feature:
    """水準 (ラグ k)"""
    return lambda g: g(v, k)


def D12(v: str, k: int = 0) -> Feature:
    """前年同月差"""
    return lambda g: g(v, k) - g(v, k + 12)


def YOY(v: str, k: int = 0) -> Feature:
    """前年同月比 (%)"""
    return lambda g: (g(v, k) / g(v, k + 12) - 1.0) * 100.0


def real_rate(country: str, k: int = 0) -> Feature:
    return lambda g: g(f"{country}.short_rate", k) - g(f"{country}.inflation", k)


@dataclass(frozen=True)
class Equation:
    key: str                      # モデルファイル名 (拡張子なし)
    target: str                   # 目的変数
    features: dict[str, Feature]  # 説明変数 (順序 = 係数の順序)
    mode: str = "level"           # "level" | "diff12" (式(2) level_from_diff)
    note: str = ""

    def target_series(self, g: Getter):
        if self.mode == "diff12":
            return g(self.target, 0) - g(self.target, 12)
        return g(self.target, 0)

    def to_level(self, yhat: float, g: Getter) -> float:
        if self.mode == "diff12":
            return g(self.target, 12) + yhat  # 式(2)
        return yhat


# ---------------------------------------------------------------- 米国ブロック (表7)
US_EQUATIONS = [
    Equation("us_unemployment_v1", "us.unemployment", {
        "米CPI(t-1)": L("us.inflation", 1),
        "米実質GDP(t-1)": L("us.gdp", 1),
    }, mode="diff12"),
    Equation("us_inflation_v1", "us.inflation", {
        "WTI原油(t-1)": L("world.oil_price", 1),
        "米製造業景況感(t-8)": L("us.manufacturing_confidence", 8),
        "米失業率(t)": L("us.unemployment", 0),
    }),
    Equation("us_wage_v1", "us.wage_growth", {
        "米失業率(t-3)": L("us.unemployment", 3),
        "米CPI(t-3)": L("us.inflation", 3),
        # ※補完: 表では「FF金利(当期)」だが FF金利は③で後から確定するため t-1 に循環カット
        "FF金利(t-1)": L("us.short_rate", 1),
    }),
    Equation("us_short_rate_v1", "us.short_rate", {
        "米CPI(t-6)": L("us.inflation", 6),
        "米失業率(t)": L("us.unemployment", 0),
        "米GDPギャップ(t-1)★循環カット": L("us.gdp_gap", 1),
    }, note="テイラールール型"),
    Equation("us_long_rate_v1", "us.long_rate", {
        "米CPI(t-1)": L("us.inflation", 1),
        "米GDPギャップ(t-1)": L("us.gdp_gap", 1),
        "FF金利(t)": L("us.short_rate", 0),
    }),
    Equation("us_manufacturing_confidence_v1", "us.manufacturing_confidence", {
        "米CPI(t)": L("us.inflation", 0),
        "米10年債 前年差(t)": D12("us.long_rate", 0),
        "WTI原油 前年比(t)": YOY("world.oil_price", 0),
    }, mode="diff12"),
    Equation("us_consumption_v1", "us.consumption", {
        "米製造業景況感(t-3)": L("us.manufacturing_confidence", 3),
        "米失業率(t)": L("us.unemployment", 0),
        "米CPI(t)": L("us.inflation", 0),
    }),
    Equation("us_real_gdp_yoy_v1", "us.gdp", {
        "米製造業景況感(t-1)": L("us.manufacturing_confidence", 1),
        "米失業率(t)": L("us.unemployment", 0),
        "米実質政策金利(t)": real_rate("us", 0),
    }),
    Equation("us_gdp_gap_v1", "us.gdp_gap", {
        "米平均時給(t-3)": L("us.wage_growth", 3),
        "米失業率(t)": L("us.unemployment", 0),
        "米CPI(t)": L("us.inflation", 0),
        "米10年債(t)": L("us.long_rate", 0),
    }),
]

# ---------------------------------------------------------------- 日本ブロック (表8)
JP_EQUATIONS = [
    Equation("jp_inflation_v1", "jp.inflation", {
        "政策金利(t)": L("jp.short_rate", 0),
        "USD/JPY(t-1)": L("jp.usd_jpy", 1),
        "WTI原油(t-6)": L("world.oil_price", 6),
        "短観DI(t-3)": L("jp.tankan_all", 3),
    }),
    Equation("jp_job_openings_v1", "jp.job_openings", {
        "景気ウォッチャー(t-3)": L("jp.economy_watcher", 3),
        "GDPギャップ(t-1)": L("jp.gdp_gap", 1),
    }),
    Equation("jp_unemployment_v2", "jp.unemployment", {
        "就業者数前年比(t-1)": L("jp.employment", 1),
        "有効求人数(t)": L("jp.job_openings", 0),
        "log(CPI+3)(t)": lambda g: np.log(g("jp.inflation", 0) + 3.0),
        "失業率(t-1) AR": L("jp.unemployment", 1),
    }),
    Equation("jp_wage_v2", "jp.wage_growth", {
        "総合CPI(t)": L("jp.inflation", 0),
        "営業利益(t-1)": L("jp.operating_profit", 1),
        "賃金上昇率(t-1) AR": L("jp.wage_growth", 1),
    }),
    Equation("jp_employment_v1", "jp.employment", {
        "営業利益(t-1)": L("jp.operating_profit", 1),
        "有効求人数(t-1)": L("jp.job_openings", 1),
        "GDPギャップ(t-1)": L("jp.gdp_gap", 1),
    }),
    Equation("jp_long_rate_v2", "jp.long_rate", {
        # ★循環カット: 求人数前年比は history[t-1] 基準 = t-1 時点の前年比
        "求人数前年比(t-1)": YOY("jp.job_openings", 1),
        "政策金利 前年差(t)": D12("jp.short_rate", 0),
        "CPI(t-1)": L("jp.inflation", 1),
        "10年債 前年差(t-1) AR": D12("jp.long_rate", 1),
    }, mode="diff12"),
    Equation("jp_usd_jpy_v2", "jp.usd_jpy", {
        "USD/JPY(t-1) AR": L("jp.usd_jpy", 1),
        "日本実質政策金利(t)": real_rate("jp", 0),
        # ※補完: 「日米金利差」は長期金利差と解釈
        "日米金利差(t)": lambda g: g("us.long_rate", 0) - g("jp.long_rate", 0),
        "日米実質金利差(t)": lambda g: (g("us.short_rate", 0) - g("us.inflation", 0))
                                    - (g("jp.short_rate", 0) - g("jp.inflation", 0)),
        "WTI原油(t)": L("world.oil_price", 0),
    }),
    Equation("jp_tankan_all_v1", "jp.tankan_all", {
        "GDPギャップ(t-1)": L("jp.gdp_gap", 1),
        "景気ウォッチャー(t-3)": L("jp.economy_watcher", 3),
        "米製造業景況感(t-6)": L("us.manufacturing_confidence", 6),
        "営業利益(t-1)★循環カット": L("jp.operating_profit", 1),
    }),
    Equation("jp_export_volume_v1", "jp.export_volume", {
        "USD/JPY(t)": L("jp.usd_jpy", 0),
        "米GDPギャップ(t-2)": L("us.gdp_gap", 2),
        "生産財在庫率(t-1)": L("jp.production_inventory", 1),
    }),
    Equation("jp_corporate_goods_price_v1", "jp.corporate_goods_price", {
        "USD/JPY(t)": L("jp.usd_jpy", 0),
        "WTI原油(t-1)": L("world.oil_price", 1),
        "米CPI(t-1)": L("us.inflation", 1),
    }),
    Equation("jp_economy_watcher_v2", "jp.economy_watcher", {
        "求人数前年比(t)": YOY("jp.job_openings", 0),
        "実質政策金利差 日-米(t)": lambda g: (g("jp.short_rate", 0) - g("jp.inflation", 0))
                                          - (g("us.short_rate", 0) - g("us.inflation", 0)),
        "賃金上昇率(t)": L("jp.wage_growth", 0),
    }),
    Equation("jp_consumer_sent_v1", "jp.consumer_sentiment", {
        "賃金上昇率(t)": L("jp.wage_growth", 0),
        "CPI(t)": L("jp.inflation", 0),
        "USD/JPY(t)": L("jp.usd_jpy", 0),
        "景気ウォッチャー(t-1)": L("jp.economy_watcher", 1),
    }),
    Equation("jp_production_inventory_v1", "jp.production_inventory", {
        "輸出数量指数(t)": L("jp.export_volume", 0),
        "WTI原油(t-3)": L("world.oil_price", 3),
        "短観DI(t-3)": L("jp.tankan_all", 3),
        "USD/JPY(t)": L("jp.usd_jpy", 0),
    }),
    Equation("jp_housing_starts_v1", "jp.housing_starts", {
        "実質賃金(t-3)": L("jp.wage_growth", 3),
        "消費者態度指数(t-1)": L("jp.consumer_sentiment", 1),
    }),
    Equation("jp_consumption_v1", "jp.consumption", {
        # 論文は「現状判断水準」と記載。状態変数は1つだけなので他の式と同じ現状判断DIを使う
        # (6.3節も同じ表現で GDPギャップ式を説明しており、そちらは方向性DIで表4に一致する)
        "景気ウォッチャー(t)": L("jp.economy_watcher", 0),
        "実質賃金(t)": L("jp.wage_growth", 0),
    }),
    Equation("jp_investment_v1", "jp.investment", {
        "景気ウォッチャー(t-3)": L("jp.economy_watcher", 3),
        "営業利益(t-1)★循環カット": L("jp.operating_profit", 1),
        "USD/JPY(t)": L("jp.usd_jpy", 0),
    }),
    Equation("jp_gdp_v1", "jp.gdp", {
        "消費者態度指数(t)": L("jp.consumer_sentiment", 0),
        "実質消費支出(t)": L("jp.consumption", 0),
    }),
    Equation("jp_gdp_gap_v1", "jp.gdp_gap", {
        "景気ウォッチャー(t-1)": L("jp.economy_watcher", 1),
        "輸出数量指数(t-1)": L("jp.export_volume", 1),
        "米GDPギャップ(t-1)": L("us.gdp_gap", 1),
    }),
    Equation("jp_operating_profit_v2", "jp.operating_profit", {
        "景気ウォッチャー(t-1)★循環カット": L("jp.economy_watcher", 1),
        "GDPギャップ(t-1)": L("jp.gdp_gap", 1),
        "10年債 前年差(t)": D12("jp.long_rate", 0),
        "営業利益(t-1) AR": L("jp.operating_profit", 1),
    }),
]

EQUATIONS: dict[str, Equation] = {e.key: e for e in US_EQUATIONS + JP_EQUATIONS}

# ---------------------------------------------------------------- 表5 計算順序
# 各要素: 方程式キー / "exog:<var>" (シナリオ入力) / "derive" (恒等式)
# 同一ステップ内の「並列」は、当期参照の依存がある場合のみ依存先を先に置いている。
STEPS: list[tuple[str, list[str]]] = [
    ("①", ["exog:world.oil_price", "exog:jp.short_rate"]),
    ("②", ["us_unemployment_v1", "us_inflation_v1", "us_wage_v1"]),
    ("③", ["us_short_rate_v1"]),
    ("④", ["us_long_rate_v1"]),
    ("⑤", ["us_manufacturing_confidence_v1", "us_consumption_v1"]),
    ("⑥", ["us_real_gdp_yoy_v1"]),
    ("⑦", ["us_gdp_gap_v1"]),
    ("⑧", ["jp_inflation_v1", "jp_job_openings_v1", "jp_unemployment_v2"]),
    ("⑨", ["jp_wage_v2"]),
    ("⑩", ["jp_employment_v1"]),
    ("⑪", ["jp_long_rate_v2"]),
    ("⑫", ["jp_usd_jpy_v2"]),
    ("⑬", ["jp_tankan_all_v1", "jp_export_volume_v1", "jp_corporate_goods_price_v1", "jp_economy_watcher_v2"]),
    ("⑭", ["jp_consumer_sent_v1", "jp_production_inventory_v1", "jp_housing_starts_v1"]),
    ("⑮", ["jp_consumption_v1"]),
    ("⑯", ["jp_investment_v1"]),
    ("⑰", ["jp_gdp_v1"]),
    ("⑱", ["jp_gdp_gap_v1"]),
    ("⑲", ["derive"]),
    ("⑳", ["jp_operating_profit_v2"]),
]

assert len(EQUATIONS) == 28
assert sorted(k for _, items in STEPS for k in items if k in EQUATIONS) == sorted(EQUATIONS)
