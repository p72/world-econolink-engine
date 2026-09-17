"""変数体系 (論文 表2・表3) と派生変数の定義。

変数名は "国.フィールド" 形式 (例: "jp.inflation")。CSV 上の列名は "." を "_" に置き換えたもの。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 表3 USState (9)
US_FIELDS = {
    "short_rate": "FF金利(%)",
    "long_rate": "10年債利回り(%)",
    "inflation": "CPI 前年比(%)",
    "gdp": "実質GDP 前年比(%)",
    "gdp_gap": "GDPギャップ(%)",
    "unemployment": "失業率(%)",
    "manufacturing_confidence": "製造業景況感 (OECD BCI)",
    "wage_growth": "実質賃金上昇率(%)",
    "consumption": "個人消費(十億ドル)",
}

# 表2 JapanState (24)
JP_FIELDS = {
    "short_rate": "政策金利(%)",
    "long_rate": "長期金利(%)",
    "term_spread": "長短金利差(%pt)",
    "risk_premium": "リスクプレミアム(%pt)",
    "inflation_premium": "インフレプレミアム(%pt)",
    "inflation": "CPI 前年比(%)",
    "unemployment": "完全失業率(%)",
    "job_openings": "有効求人数(万件)",
    "employment": "就業者数 前年比(%)",
    "wage_growth": "実質賃金上昇率(%)",
    "usd_jpy": "USD/JPY",
    "tankan_all": "短観 全産業業況判断DI",
    "economy_watcher": "景気ウォッチャー 現状判断DI(季調)",
    "consumer_sentiment": "消費者態度指数",
    "export_volume": "輸出数量指数",
    "production_inventory": "生産財在庫率指数",
    "corporate_goods_price": "企業物価指数",
    "housing_starts": "新設住宅着工床面積 前年比(%)",
    "consumption": "実質消費支出(兆円)",
    "investment": "実質設備投資 前年比(%)",
    "gdp": "実質GDP 前年比(%)",
    "gdp_gap": "GDPギャップ(%)",
    "gdp_per_worker": "1人当たりGDP 前年比(%)",
    "operating_profit": "営業利益 (CI一致 C8)",
}

WORLD_FIELDS = {"oil_price": "WTI原油($/bbl)"}

ALL_VARS: dict[str, str] = {
    **{f"world.{k}": v for k, v in WORLD_FIELDS.items()},
    **{f"us.{k}": v for k, v in US_FIELDS.items()},
    **{f"jp.{k}": v for k, v in JP_FIELDS.items()},
}

# モデルを持たず恒等式で決まる変数 (データ側にも同じ式で作る)
DERIVED = ["jp.term_spread", "jp.inflation_premium", "jp.risk_premium", "jp.gdp_per_worker"]


def col(var: str) -> str:
    return var.replace(".", "_")


def derive(v: dict) -> dict:
    """派生変数を計算して返す。v は float の dict でも pd.Series の dict でもよい。

    - term_spread = long − short
    - inflation_premium = inflation − short  (期待実質短期金利ギャップの代理; 論文に定義が無いため本実装の仮定)
    - risk_premium = term_spread − inflation_premium
    - gdp_per_worker ≈ 実質GDP前年比 − 就業者数前年比 (⑲ モデルなしの状態更新)
    """
    out = {}
    out["jp.term_spread"] = v["jp.long_rate"] - v["jp.short_rate"]
    out["jp.inflation_premium"] = v["jp.inflation"] - v["jp.short_rate"]
    out["jp.risk_premium"] = out["jp.term_spread"] - out["jp.inflation_premium"]
    out["jp.gdp_per_worker"] = v["jp.gdp"] - v["jp.employment"]
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = ["jp.long_rate", "jp.short_rate", "jp.inflation", "jp.gdp", "jp.employment"]
    if all(col(n) in df for n in needed):
        vals = derive({n: df[col(n)] for n in needed})
        for k, s in vals.items():
            df[col(k)] = s
    return df


def empty_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=[col(v) for v in ALL_VARS])
