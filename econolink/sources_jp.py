"""日本ブロックの実データ取得: e-Stat + 日銀時系列統計API + 内閣府 (GDP・GDPギャップ)。

論文 4.1 の「e-Stat 以外の政府公表データ (日銀・内閣府・厚労省・財務省)」もここで自動取得する。すべて公式統計そのもので、代理系列は使わない。
- e-Stat DB (getStatsData) で最新まで取れる統計はそこから取る
- DB が古い/非対応の統計 (毎月勤労統計・一般職業紹介状況・建築着工統計) は、e-Stat の公表 Excel をカタログ等から最新版を探して読む
- e-Stat に無い統計 (短観・企業物価・GDP・GDPギャップ) は日銀・内閣府の公表元から取る
"""
from __future__ import annotations

import gzip
import html
import io
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from . import estat

UA = {"User-Agent": "Mozilla/5.0"}

# 変数 → 出典の説明 (README・ログ表示用)
SOURCES = {
    "jp_inflation": "e-Stat 消費者物価指数(2020年基準) 総合 前年同月比 [0003427113]",
    "jp_unemployment": "労働力調査 完全失業率 季節調整値 (e-Stat 景気動向指数 Lg6 収録値) [0003446462]",
    "jp_job_openings": "一般職業紹介状況 有効求人数 (パート含む一般) 季節調整値 万人 (e-Stat 長期時系列表_6)",
    "jp_employment": "労働力調査 就業者数 → 前年同月比 [0003005798]",
    "jp_wage_growth": "毎月勤労統計 実質賃金 現金給与総額 5人以上 調査産業計 前年同月比 公表値 (長期時系列表25-1 + 最新速報)",
    "jp_economy_watcher": "景気ウォッチャー調査 現状判断DI(方向性) 季節調整値 合計・全国 [0003348423]",
    "jp_consumer_sentiment": "消費動向調査 消費者態度指数 季節調整値 (e-Stat 景気動向指数 L6 収録値)",
    "jp_export_volume": "輸出数量指数 季節調整値 (e-Stat 景気動向指数 C10 収録値)",
    "jp_production_inventory": "鉱工業用生産財在庫率指数 (e-Stat 景気動向指数 L2 収録値)",
    "jp_housing_starts": "建築着工統計 新設住宅着工床面積 総計 前年同月比 公表値 (住宅着工統計 時系列表 月次)",
    "jp_consumption": "内閣府 QE 実質民間最終消費支出 季節調整系列 年率 兆円 四半期→月次ffill",
    "jp_investment": "実質法人企業設備投資(全産業) → 前年同月比 (e-Stat 景気動向指数 Lg3 収録値)",
    "jp_operating_profit": "営業利益(全産業) (e-Stat 景気動向指数 C8 収録値)",
    "jp_tankan_all": "日銀 短観 業況判断DI 全規模全産業 実績 [CO/TK99F0000601GCQ00000] 四半期→月次ffill",
    "jp_corporate_goods_price": "日銀 国内企業物価指数 総平均(2020年=100) [PR01/PRCG20_2200000000]",
    "jp_short_rate": "日銀 無担保コールレート O/N 月平均 [FM02/STRACLUCON]",
    "jp_usd_jpy": "日銀 東京市場 ドル・円 スポット 17時時点 月中平均 [FM08/FXERM07]",
    "jp_long_rate": "財務省 国債金利情報 10年 (日次) の月平均",
    "jp_gdp": "内閣府 QE 実質GDP 季節調整系列 → 前年同期比 四半期→月次ffill",
    "jp_gdp_gap": "内閣府 月例経済報告 GDPギャップ 四半期→月次ffill",
}

CI_CODES = {  # 景気動向指数 個別系列 (cat01)。各統計の公表値がそのまま収録されている
    "2080": "op_profit", "1020": "prod_inventory", "1060": "consumer_sentiment",
    "2100": "export_volume", "3030": "investment", "3060": "unemployment",
}


def _yoy(s: pd.Series) -> pd.Series:
    return (s / s.shift(12) - 1) * 100


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def _quarterly_to_monthly(s: pd.Series) -> pd.Series:
    """四半期値 (期初月 index) を月次に前方補間 (論文 5.1)。期末の2か月だけ埋める。"""
    idx = pd.date_range(s.index.min(), s.index.max() + pd.DateOffset(months=2), freq="MS")
    return s.reindex(idx).ffill(limit=2)


# ------------------------------------------------------------------ e-Stat
def fetch_estat() -> pd.DataFrame:
    print("  e-Stat: 景気動向指数 ...")
    ci = estat.monthly_frame("0003446462", by="cat01", cdCat01=",".join(CI_CODES)).rename(columns=CI_CODES)
    print("  e-Stat: 消費者物価指数 ...")
    cpi = estat.monthly_series("0003427113", cdTab="3", cdCat01="0001", cdArea="00000")
    print("  e-Stat: 労働力調査 就業者数 ...")
    employed = estat.monthly_series("0003005798", cdCat01="000", cdCat02="02", cdCat03="0")
    print("  e-Stat: 景気ウォッチャー調査 ...")
    # 現状判断DI (方向性) 季節調整値 = 公表の中心指標。
    # 水準DI (0003348427 cat02=120) より、論文 表4 の係数に近い推定結果になることを確認済み。
    watcher = estat.monthly_series("0003348423", cdCat01="100", cdCat02="100")
    print("  e-Stat: 一般職業紹介状況 有効求人数 (公表Excel) ...")
    job_openings = fetch_job_openings()
    print("  e-Stat: 建築着工統計 住宅着工床面積 (公表Excel) ...")
    housing = fetch_housing_starts_yoy()
    print("  e-Stat/厚労省: 毎月勤労統計 実質賃金 (公表Excel) ...")
    wage = fetch_real_wage_yoy()
    return pd.DataFrame({
        "jp_inflation": cpi,
        "jp_unemployment": ci["unemployment"],
        "jp_job_openings": job_openings,
        "jp_employment": _yoy(employed),
        "jp_wage_growth": wage,
        "jp_economy_watcher": watcher,
        "jp_consumer_sentiment": ci["consumer_sentiment"],
        "jp_export_volume": ci["export_volume"],
        "jp_production_inventory": ci["prod_inventory"],
        "jp_housing_starts": housing,
        "jp_investment": _yoy(ci["investment"]),
        "jp_operating_profit": ci["op_profit"],
    })


_ZEN2HAN = str.maketrans("０１２３４５６７８９", "0123456789")


def fetch_job_openings() -> pd.Series:
    """一般職業紹介状況 長期時系列表_6 有効求人数 (パート含む一般=除学卒) 季節調整値, 万人。

    この統計は getStatsData 非対応のため、e-Stat の公表 Excel をカタログから探して読む。
    """
    import openpyxl

    f = estat.latest_catalog_file("00450222", r"^長期時系列表_6_有効求人数", exclude_dataset="旧様式")
    wb = openpyxl.load_workbook(io.BytesIO(estat.download(f["url"])), data_only=True, read_only=True)
    ws = next(w for w in wb.worksheets if "パート含む" in w.title)
    rows = list(ws.iter_rows(values_only=True))
    kind = next(r for r in rows if r and "実数" in r)   # 「実数」「季節調整値」の区分行
    header = rows[rows.index(kind) + 1]                 # 西暦, 和暦, １月, ２月 ...
    month_cols = {}
    for j, (k, h) in enumerate(zip(kind, header)):
        m = re.fullmatch(r"(\d{1,2})月", str(h or "").translate(_ZEN2HAN))
        if k == "季節調整値" and m:
            month_cols[int(m.group(1))] = j
    if len(month_cols) != 12:
        raise RuntimeError(f"有効求人数: 季節調整値の月列が特定できません ({f['url']})")
    out = {}
    for r in rows:
        y = re.fullmatch(r"(\d{4})年", str(r[0] or ""))
        if y:
            for month, j in month_cols.items():
                if isinstance(r[j], (int, float)):
                    out[pd.Timestamp(int(y.group(1)), month, 1)] = r[j] / 10000.0
    return pd.Series(out, dtype=float).sort_index()


def fetch_housing_starts_yoy() -> pd.Series:
    """建築着工統計 住宅着工統計 時系列表(月次) 新設住宅 総計 床面積の前年同月比 (%, 公表値)。

    列: A=期間, B=総計戸数, C=戸数前年比, D=総計床面積(千m2), E=床面積前年比。
    期間は1月の行だけ元号付き (例「R８年 １月」)、他の月は月の数字のみ。旧形式 .xls のため xlrd で読む。
    """
    import unicodedata

    import xlrd

    f = estat.latest_catalog_file("00600120", r"利用関係別.*戸数", dataset_pattern=r"^住宅着工統計_時系列表_月次",
                                  search_word="住宅着工統計 時系列表 月次")
    sheet = xlrd.open_workbook(file_contents=estat.download(f["url"])).sheet_by_index(0)
    header = "".join(str(v) for r in range(8) for v in sheet.row_values(r)[1:5])
    if "総計" not in header or "床面積" not in header:
        raise RuntimeError(f"住宅着工統計: 列のレイアウトが変わっています ({f['url']})")
    era = {"S": 1925, "H": 1988, "R": 2018}
    out, year, started = {}, None, False
    for i in range(sheet.nrows):
        label = unicodedata.normalize("NFKC", str(sheet.cell_value(i, 0))).strip()
        if m := re.search(r"([SHR])\s*(\d+|元)\s*年\s*(\d+)\s*月", label):
            year = era[m.group(1)] + (1 if m.group(2) == "元" else int(m.group(2)))
            month, started = int(m.group(3)), True
        elif started and re.fullmatch(r"\d{1,2}(\.0)?", label):
            month = int(float(label))
        elif started:
            break  # 月次ブロックの終わり
        else:
            continue
        v = sheet.cell_value(i, 4)
        if v != "":
            out[pd.Timestamp(year, month, 1)] = float(v)
    return pd.Series(out, dtype=float).sort_index()


MAIKIN_LIST = ("https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00450071&tstat=000001011791"
               "&cycle=0&tclass1=000001035519&tclass2=000001144287&layout=datalist&tclass3val=0")
MAIKIN_T25_FALLBACK = "000032189738"  # 表番号25-1 (2026-09-17 確認)


def _maikin_t25_url() -> str:
    """毎月勤労統計 長期時系列表 表番号25-1 (実質賃金・現金給与総額・5人以上) の Excel URL を一覧ページから探す。"""
    try:
        page = _fetch(MAIKIN_LIST).decode("utf-8", "replace")
        for block in page.split('<article class="stat-dataset_list-item">')[1:]:
            text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", block)))
            if "表番号 25-1 " in text and "実質賃金（現金給与総額）" in text and "５人以上" in text:
                if m := re.search(r"statInfId=(\d+)&(?:amp;)?fileKind=4", block):
                    sid = m.group(1)
                    break
        else:
            sid = MAIKIN_T25_FALLBACK
    except Exception:
        sid = MAIKIN_T25_FALLBACK
    return f"https://www.e-stat.go.jp/stat-search/file-download?statInfId={sid}&fileKind=4"


def _real_wage_yoy_confirmed() -> pd.Series:
    """確報の実質賃金 前年同月比 (%)。シート TL=調査産業計、前年比ブロックの 8..19 列が 1〜12月。"""
    df = pd.read_excel(io.BytesIO(_fetch(_maikin_t25_url())), sheet_name="TL", header=None)
    start = df[0].astype(str).str.contains("前年比").idxmax() + 3
    if str(df.iat[start - 2, 8]).strip() != "1" or str(df.iat[start - 2, 19]).strip() != "12":
        raise RuntimeError("毎月勤労統計 表25-1: 月列のレイアウトが変わっています")
    out = {}
    for i in range(start + 1, len(df)):
        year = str(df.iat[i, 0]).strip()
        if not year.isdigit():
            if out:
                break
            continue
        for month in range(1, 13):
            v = pd.to_numeric(df.iat[i, 7 + month], errors="coerce")  # "-" は欠損
            if pd.notna(v):
                out[pd.Timestamp(int(year), month, 1)] = float(v)
    return pd.Series(out, dtype=float).sort_index()


def _real_wage_yoy_latest() -> pd.Series:
    """厚労省の最新月次公表 (速報を含む) 時系列第1表から、現金給与総額の実質前年比 (直近14か月)。"""
    page = _fetch("https://www.mhlw.go.jp/toukei/list/30-1a.html").decode("utf-8", "replace")
    releases = re.findall(r"/toukei/itiran/roudou/monthly/(r\d\d)/(\d{4})([pr])/\2\3\.html", page)
    era, yymm, kind = max(releases, key=lambda x: (x[1], x[2] == "r"))  # 同月なら確報(r)優先
    url = f"https://www.mhlw.go.jp/toukei/itiran/roudou/monthly/{era}/{yymm}{kind}/xls/{yymm}t01{kind}.xlsx"
    df = pd.read_excel(io.BytesIO(_fetch(url)), header=None)
    out, year, in_block = {}, None, False
    for i in range(len(df)):
        label = str(df.iat[i, 0]).replace("　", "").strip().translate(_ZEN2HAN)
        if re.sub(r"\s", "", label) == "現金給与総額":
            in_block = True
            continue
        if not in_block:
            continue
        if label and label != "nan" and not re.search(r"\d", label):
            break  # 次の区分 (きまって支給する給与 等)
        m = re.match(r"(?:(\d{4})年)?(\d{1,2})月", label)
        if not m:
            continue
        year = int(m.group(1)) if m.group(1) else year
        v = pd.to_numeric(df.iat[i, 4], errors="coerce")  # 4列目 = 実質前年比
        if pd.notna(v):
            out[pd.Timestamp(year, int(m.group(2)), 1)] = float(v)
    return pd.Series(out, dtype=float).sort_index()


def fetch_real_wage_yoy() -> pd.Series:
    """毎月勤労統計 実質賃金 (現金給与総額, 5人以上, 調査産業計) 前年同月比 %。

    指数からの自前計算ではなく公表値を使う: 公表の増減率はサンプル入替・ベンチマーク更新の段差を
    補正した「接続」ベースで計算されており、指数比とは最大3ポイント以上異なる月がある。
    """
    confirmed = _real_wage_yoy_confirmed()
    try:
        return _real_wage_yoy_latest().combine_first(confirmed)  # 重なる月は最新公表を優先
    except Exception as e:
        print(f"  (警告) 毎月勤労統計の最新速報を取得できず確報のみ使用: {e}")
        return confirmed


# ------------------------------------------------------------------ 日銀
def boj_series(db: str, code: str, freq: str = "M", start: str = "200001") -> pd.Series:
    params = {"format": "json", "lang": "jp", "db": db, "code": code, "startDate": start}
    req = urllib.request.Request("https://www.stat-search.boj.or.jp/api/v1/getDataCode?"
                                 + urllib.parse.urlencode(params), headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    js = json.loads(raw.decode("utf-8"))
    if js["STATUS"] != 200:
        raise RuntimeError(f"BOJ {db}/{code}: {js['MESSAGEID']} {js['MESSAGE']}")
    vals = js["RESULTSET"][0]["VALUES"]
    idx = []
    for d in map(str, vals["SURVEY_DATES"]):  # 月次 YYYYMM / 四半期 YYYYQQ
        y, p = int(d[:4]), int(d[4:])
        idx.append(pd.Timestamp(y, (p - 1) * 3 + 1 if freq == "Q" else p, 1))
    return pd.Series(pd.to_numeric(vals["VALUES"], errors="coerce"), index=idx, name=code).dropna()


def fetch_boj() -> pd.DataFrame:
    print("  日銀: 短観・企業物価指数・コールレート・ドル円 ...")
    tankan = boj_series("CO", "TK99F0000601GCQ00000", freq="Q", start="200001"); time.sleep(1.5)
    cgpi = boj_series("PR01", "PRCG20_2200000000"); time.sleep(1.5)
    call = boj_series("FM02", "STRACLUCON"); time.sleep(1.5)
    usd_jpy = boj_series("FM08", "FXERM07")
    return pd.DataFrame({
        "jp_tankan_all": _quarterly_to_monthly(tankan),
        "jp_corporate_goods_price": cgpi,
        "jp_short_rate": call,
        "jp_usd_jpy": usd_jpy,
    })


# ------------------------------------------------------------------ 財務省
def _wareki_to_date(label: str) -> pd.Timestamp | None:
    m = re.fullmatch(r"([SHR])(\d+|元)\.(\d+)\.(\d+)", label.strip())
    if not m:
        return None
    base = {"S": 1925, "H": 1988, "R": 2018}[m.group(1)]
    return pd.Timestamp(base + (1 if m.group(2) == "元" else int(m.group(2))), int(m.group(3)), int(m.group(4)))


def fetch_jgb10y_monthly() -> pd.Series:
    """財務省 国債金利情報 10年 (日次) の月平均。過去分 jgbcm_all.csv + 当月分 jgbcm.csv。

    当月分は月の途中の平均になるため、今日を含む月は除外する。
    """
    base = "https://www.mof.go.jp/jgbs/reference/interest_rate/"
    daily = {}
    for url in [base + "data/jgbcm_all.csv", base + "jgbcm.csv"]:
        df = pd.read_csv(io.StringIO(_fetch(url).decode("cp932")), skiprows=1, dtype=str)
        for label, value in zip(df["基準日"], df["10年"]):
            date = _wareki_to_date(str(label))
            v = pd.to_numeric(value, errors="coerce")  # 未発行期間は "-"
            if date is not None and pd.notna(v):
                daily[date] = float(v)
    s = pd.Series(daily).sort_index()
    monthly = s.groupby(s.index.to_period("M")).mean()
    monthly.index = monthly.index.to_timestamp()
    return monthly[monthly.index < pd.Timestamp.today().to_period("M").to_timestamp()]


# ------------------------------------------------------------------ 内閣府
def _latest_qe_csv(series: str = "gaku-jk") -> str:
    """QE の CSV は公表ごとに URL が変わるため、メニューから最新 (年, 四半期, 1次/2次) を探す。"""
    html = _fetch("https://www.esri.cao.go.jp/jp/sna/menu.html").decode("utf-8", "replace")
    pat = rf'href="(/jp/sna/data/data_list/sokuhou/files/(\d{{4}})/qe(\d{{2}})(\d)(_2)?/tables/{series}\d{{4}}\.csv)"'
    cands = [(int(y), int(q), 2 if rev else 1, path) for path, y, _yy, q, rev in re.findall(pat, html)]
    if not cands:
        raise RuntimeError("ESRI: QE の CSV リンクが見つかりません")
    return "https://www.esri.cao.go.jp" + max(cands)[3]


def qe_real_sa() -> pd.DataFrame:
    """QE 実質季節調整系列 (2020暦年連鎖価格, 10億円, 年率) の GDP と 民間最終消費支出。"""
    df = pd.read_csv(io.StringIO(_fetch(_latest_qe_csv()).decode("cp932")), header=None, dtype=str)
    header = df.iloc[:7].fillna("").astype(str)
    col_of = {name: next(j for j in df.columns if name in "".join(header[j]))
              for name in ["国内総生産", "民間最終消費支出"]}
    months = {"1- 3": 1, "4- 6": 4, "7- 9": 7, "10-12": 10}
    year, rows = None, {}
    for _, r in df.iterrows():
        m = re.match(r"^(?:(\d{4})/)?\s*(1- 3|4- 6|7- 9|10-12)\.$", str(r[0]).strip())
        if not m:
            continue
        year = int(m.group(1)) if m.group(1) else year
        rows[pd.Timestamp(year, months[m.group(2)], 1)] = {
            name: float(str(r[j]).replace(",", "")) for name, j in col_of.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def cao_gdp_gap() -> pd.Series:
    index_url = "https://www5.cao.go.jp/keizai3/getsurei/getsurei-index.html"
    m = re.search(r'href="([^"]*\d{3,4}gap\.xlsx)"', _fetch(index_url).decode("utf-8", "replace"))
    if not m:
        raise RuntimeError("内閣府: GDPギャップの Excel リンクが見つかりません")
    df = pd.read_excel(io.BytesIO(_fetch(urllib.parse.urljoin(index_url, m.group(1)))),
                       sheet_name="四半期", header=None)
    quarters = {"Ⅰ": 1, "Ⅱ": 4, "Ⅲ": 7, "Ⅳ": 10}
    year, rows = None, {}
    for _, r in df.iloc[6:].iterrows():
        if pd.notna(r[0]) and str(r[0]).strip().isdigit():
            year = int(str(r[0]).strip())
        q = str(r[1]).strip()
        if year is not None and q in quarters:
            rows[pd.Timestamp(year, quarters[q], 1)] = pd.to_numeric(r[2], errors="coerce")
    return pd.Series(rows).sort_index().dropna()


def fetch_cao() -> pd.DataFrame:
    print("  内閣府: QE (実質GDP・実質民間最終消費支出)・GDPギャップ ...")
    qe = qe_real_sa()
    return pd.DataFrame({
        "jp_gdp": _quarterly_to_monthly(qe["国内総生産"].pct_change(4) * 100),
        "jp_consumption": _quarterly_to_monthly(qe["民間最終消費支出"] / 1000),  # 10億円 → 兆円
        "jp_gdp_gap": _quarterly_to_monthly(cao_gdp_gap()),
    })


def fetch_japan(start: str = "2000-01-01") -> pd.DataFrame:
    print("  財務省: 国債金利 10年 ...")
    jgb = fetch_jgb10y_monthly().rename("jp_long_rate")
    parts = [fetch_estat(), fetch_boj(), fetch_cao(), jgb]
    df = pd.concat(parts, axis=1).sort_index().loc[start:]
    df.index.name = "date"
    return df
