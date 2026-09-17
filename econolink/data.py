"""データ基盤 (論文 第4章)。

- make_synthetic(): 実データが揃っていなくても全パイプラインを動かすための合成データ (デモ用・経済的意味はない)
- fetch_us():       米国9系列 + 原油 (FRED CSV・BLS API, いずれもキー不要)
- load_dataset():   data/*.csv を日付キーでマージ (後勝ち)。日本の e-Stat 系列は手動CSVで追加する
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .variables import ALL_VARS, DERIVED, col


# ============================================================ 合成データ
def make_synthetic(start: str = "2004-01-01", end: str = "2026-02-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="MS")
    n = len(idx)

    def ar1(phi, sd, x0=0.0):
        x = np.empty(n)
        x[0] = x0
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal(0, sd)
        return x

    def lag(a, k):
        return np.r_[np.full(k, a[0]), a[:-k]] if k else a

    # 景気循環ファクター (AR(2))
    cu = np.zeros(n)
    for t in range(2, n):
        cu[t] = 1.75 * cu[t - 1] - 0.8 * cu[t - 2] + rng.normal(0, 0.12)
    cu /= cu.std()
    cj = 0.6 * lag(cu, 2) + 0.5 * ar1(0.9, 0.3)
    cj /= cj.std()

    oil = 70 * np.exp(0.35 * ar1(0.98, 0.06))

    # ---- US
    oil_yoy = np.nan_to_num((oil / lag(oil, 12) - 1) * 100)
    us_gap = 1.5 * cu + ar1(0.6, 0.15)
    us_unemp = 5.0 - 1.1 * lag(cu, 2) + ar1(0.97, 0.05)
    us_mfg = 5.0 * cu + 0.03 * oil_yoy + ar1(0.7, 1.0)
    us_infl = 2.3 + 0.7 * lag(cu, 6) + 0.02 * (lag(oil, 1) - 70) + ar1(0.85, 0.15)
    us_short = np.clip(2.5 + 1.2 * (lag(us_infl, 6) - 2.3) - 0.5 * (us_unemp - 5) + 0.5 * lag(us_gap, 1)
                       + ar1(0.5, 0.1), 0.05, None)
    # 金融政策の波及: 実質金利が高いほど成長率が下がる (これが無いと失業率⇔GDPのループが発散しやすい)
    us_gdp = 2.0 + 1.8 * lag(cu, 1) - 0.6 * ((us_short - us_infl) - 0.3) + ar1(0.5, 0.25)
    us_long = us_short * 0.6 + 1.8 + 0.2 * lag(us_infl, 1) + ar1(0.95, 0.08)
    us_wage = 1.0 - 0.3 * (lag(us_unemp, 3) - 5) - 0.3 * (lag(us_infl, 3) - 2.3) + ar1(0.6, 0.2)
    us_cons = 14000 + 150 * lag(us_mfg, 3) - 400 * (us_unemp - 5) - 100 * us_infl + ar1(0.8, 40)

    # ---- JP
    jp_short = np.where(idx < "2016-02-01", 0.1, -0.1)
    jp_short = np.where(idx >= "2024-03-01", 0.1, jp_short)
    jp_short = np.where(idx >= "2024-08-01", 0.25, jp_short)
    jp_short = np.where(idx >= "2025-02-01", 0.5, jp_short)
    usd_jpy = 110 + 6 * (us_long - 3.0) + 12 * ar1(0.98, 0.12)
    jp_infl = 0.5 + 0.4 * lag(cj, 6) + 0.02 * (lag(oil, 6) - 70) + 0.04 * (lag(usd_jpy, 1) - 110) + ar1(0.8, 0.12)
    jp_long = 0.3 + 0.8 * jp_short + 0.25 * lag(cj, 1) + 0.1 * jp_infl + ar1(0.97, 0.04)
    ew = 48 + 5 * cj + ar1(0.5, 1.2)
    tankan = 5 + 9 * lag(cj, 1) + 0.3 * lag(us_mfg, 6) + ar1(0.7, 1.5)
    job_open = 200 + 25 * lag(cj, 2) + 20 * ar1(0.98, 0.1)
    jp_gap = -0.4 + 1.2 * lag(cj, 1) + 0.3 * lag(us_gap, 1) + ar1(0.6, 0.15)
    op_profit = 100 + 8 * lag(cj, 1) + 5 * ar1(0.95, 0.2)
    employment = 0.3 + 0.4 * lag(cj, 1) + ar1(0.7, 0.1)
    unemp = 3.5 - 0.5 * cj + ar1(0.97, 0.03)
    wage = 0.2 + 0.4 * lag(cj, 1) - 0.4 * (jp_infl - 0.5) + ar1(0.6, 0.2)
    export_vol = 100 + 6 * cj + 0.15 * (usd_jpy - 110) + 2 * lag(us_gap, 2) + ar1(0.8, 1.0)
    cgpi = 100 + 0.2 * (usd_jpy - 110) + 0.12 * (lag(oil, 1) - 70) + 0.8 * lag(us_infl, 1) + ar1(0.9, 0.3)
    sentiment = 38 + 3 * cj + 0.5 * wage - 0.6 * jp_infl + 0.2 * lag(ew - 48, 1) + ar1(0.7, 0.6)
    prod_inv = 105 - 0.4 * (export_vol - 100) + 0.05 * (lag(oil, 3) - 70) - 0.2 * lag(tankan, 3) + ar1(0.7, 1.5)
    housing = 1.5 * lag(wage, 3) + 0.8 * (lag(sentiment, 1) - 38) + ar1(0.3, 3.0)
    consumption = 25 + 0.08 * (ew - 48) + 0.2 * wage + ar1(0.8, 0.15)
    investment = 2 + 0.5 * (lag(ew, 3) - 48) + 0.2 * (lag(op_profit, 1) - 100) + ar1(0.6, 1.2)
    gdp = 0.8 + 0.2 * (sentiment - 38) + 1.5 * (consumption - 25) + ar1(0.6, 0.3)

    v = {
        "world.oil_price": oil,
        "us.short_rate": us_short, "us.long_rate": us_long, "us.inflation": us_infl, "us.gdp": us_gdp,
        "us.gdp_gap": us_gap, "us.unemployment": us_unemp, "us.manufacturing_confidence": us_mfg,
        "us.wage_growth": us_wage, "us.consumption": us_cons,
        "jp.short_rate": jp_short, "jp.long_rate": jp_long, "jp.inflation": jp_infl,
        "jp.unemployment": unemp, "jp.job_openings": job_open, "jp.employment": employment,
        "jp.wage_growth": wage, "jp.usd_jpy": usd_jpy, "jp.tankan_all": tankan,
        "jp.economy_watcher": ew, "jp.consumer_sentiment": sentiment, "jp.export_volume": export_vol,
        "jp.production_inventory": prod_inv, "jp.corporate_goods_price": cgpi,
        "jp.housing_starts": housing, "jp.consumption": consumption, "jp.investment": investment,
        "jp.gdp": gdp, "jp.gdp_gap": jp_gap, "jp.operating_profit": op_profit,
    }
    df = pd.DataFrame({col(k): a for k, a in v.items()}, index=idx)
    df.index.name = "date"
    return df.round(4)


# ============================================================ FRED (APIキー不要)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _fred(series_id: str) -> pd.Series:
    with urllib.request.urlopen(FRED_URL.format(series_id), timeout=60) as r:
        s = pd.read_csv(io.BytesIO(r.read()), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
    return s.astype(float)


def _quarterly_to_monthly(s: pd.Series) -> pd.Series:
    """四半期値 (期初月 index) を月次に前方補間 (論文 5.1)。各四半期の3か月だけ埋める。"""
    idx = pd.date_range(s.index.min(), s.index.max() + pd.DateOffset(months=2), freq="MS")
    return s.reindex(idx).ffill(limit=2)


def _yoy(s: pd.Series) -> pd.Series:
    return (s / s.shift(12) - 1) * 100


def fetch_bls(series_id: str, start_year: int = 2000) -> pd.Series:
    """BLS Public Data API v2 (登録なし: 1リクエスト10年分・1日25リクエストまで)。"""
    end_year, rows = pd.Timestamp.today().year, {}
    for y0 in range(start_year, end_year + 1, 10):
        body = json.dumps({"seriesid": [series_id], "startyear": str(y0),
                           "endyear": str(min(y0 + 9, end_year))}).encode()
        req = urllib.request.Request("https://api.bls.gov/publicAPI/v2/timeseries/data/", data=body,
                                     headers={"Content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            js = json.loads(r.read())
        if js["status"] != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS {series_id}: {js['status']} {js.get('message')}")
        for x in js["Results"]["series"][0]["data"]:
            if x["period"].startswith("M") and x["period"] != "M13":  # M13 = 年平均
                # 公表されなかった月は "-" (例: 政府機関閉鎖で調査が行われなかった月)
                rows[pd.Timestamp(int(x["year"]), int(x["period"][1:]), 1)] = pd.to_numeric(x["value"], errors="coerce")
        time.sleep(1.0)
    return pd.Series(rows, dtype=float).sort_index()


# 変数 → 出典 (すべて各国統計機関の公表値。FRED は配布経路としてのみ使用)
US_SOURCES = {
    "world_oil_price": "EIA WTI原油スポット価格 月平均 (FRED MCOILWTICO)",
    "us_short_rate": "FRB H.15 フェデラルファンド実効金利 月平均 (FRED FEDFUNDS)",
    "us_long_rate": "FRB H.15 10年国債利回り(固定満期) 月平均 (FRED GS10)",
    "us_inflation": "BLS CPI-U 全項目 季節調整前 → 前年同月比 = BLS公表の12か月変化率 (FRED CPIAUCNS)",
    "us_unemployment": "BLS 失業率 季節調整値 (FRED UNRATE)",
    "us_gdp": "BEA 実質GDP 前年同期比 公表値 (FRED A191RO1Q156NBEA) 四半期→月次ffill",
    "us_gdp_gap": "BEA 実質GDP (GDPC1) と CBO 潜在GDP (GDPPOT) の乖離率 (論文 4.1 の算出方法) 四半期→月次ffill",
    "us_manufacturing_confidence": "OECD 企業景況感 製造業 米国 (FRED BSCICP02USM460S, 論文指定の系列)",
    "us_wage_growth": "BLS Real Earnings 実質平均時給(民間計, 1982-84年ドル) → 前年同月比 (BLS API CES0500000013)",
    "us_consumption": "BEA 実質個人消費支出 季節調整・年率 十億ドル(連鎖2017年) (FRED PCEC96)",
}


def fetch_us(start: str = "2000-01-01") -> pd.DataFrame:
    print("米国系列を取得中 (FRED / BLS) ...")
    raw = {sid: _fred(sid) for sid in [
        "FEDFUNDS", "GS10", "CPIAUCNS", "UNRATE", "A191RO1Q156NBEA", "GDPC1", "GDPPOT",
        "BSCICP02USM460S", "PCEC96", "MCOILWTICO",
    ]}
    last = raw["UNRATE"].index.max()  # 潜在GDPは将来まで入っているので月次統計の最終月で切る
    df = pd.DataFrame({
        "world_oil_price": raw["MCOILWTICO"],
        "us_short_rate": raw["FEDFUNDS"],
        "us_long_rate": raw["GS10"],
        "us_inflation": _yoy(raw["CPIAUCNS"]),
        "us_unemployment": raw["UNRATE"],
        "us_gdp": _quarterly_to_monthly(raw["A191RO1Q156NBEA"]),
        "us_gdp_gap": _quarterly_to_monthly((raw["GDPC1"] / raw["GDPPOT"] - 1).dropna() * 100),
        "us_manufacturing_confidence": raw["BSCICP02USM460S"],
        "us_wage_growth": _yoy(fetch_bls("CES0500000013", start_year=int(start[:4]))),
        "us_consumption": raw["PCEC96"],
    })
    df = df.loc[start:last]
    df.index.name = "date"
    return df


# ============================================================ 読み込み
def load_dataset(data_dir: Path, files: list[str] | None = None) -> pd.DataFrame:
    """data_dir 内の CSV を順にマージ。後のファイルの非欠損値が優先される。"""
    paths = [data_dir / f for f in files] if files else sorted(data_dir.glob("*.csv"))
    merged: pd.DataFrame | None = None
    for p in paths:
        if p.name == "template_jp_manual.csv":
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = df.index.to_period("M").to_timestamp()
        merged = df if merged is None else _overlay(merged, df)
    if merged is None:
        raise FileNotFoundError(f"{data_dir} に CSV がありません")
    known = [col(v) for v in ALL_VARS]
    return merged.reindex(columns=[c for c in known if c in merged.columns]).sort_index()


def _overlay(base: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    out = base.reindex(index=base.index.union(top.index), columns=base.columns.union(top.columns))
    out.update(top)
    return out


def write_template(path: Path) -> None:
    """日本の e-Stat 等の系列を手入力するための空テンプレート。"""
    cols = [col(v) for v in ALL_VARS if v.startswith("jp.") and v not in DERIVED]
    pd.DataFrame(columns=["date"] + cols).to_csv(path, index=False, encoding="utf-8-sig")
