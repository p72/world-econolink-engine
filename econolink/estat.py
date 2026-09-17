"""e-Stat API (getStatsData) から日本の月次系列を取得する。

appId は環境変数 ESTAT_APP_ID から読む (https://www.e-stat.go.jp/api/ で無料発行)。
連続アクセスで 403 が返ることがあるため、リクエスト間隔を空け、403 時は待ってリトライする。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

API = "https://api.e-stat.go.jp/rest/3.0/app/json/"
BASE = API + "getStatsData"
UA = {"User-Agent": "Mozilla/5.0"}
SLEEP, RETRY_SLEEP, MAX_RETRIES = 2.0, 20.0, 2
_last = 0.0


def _app_id() -> str:
    app_id = os.environ.get("ESTAT_APP_ID")
    if not app_id:
        raise RuntimeError("環境変数 ESTAT_APP_ID に e-Stat の appId を設定してください")
    return app_id


def _get(url: str) -> bytes:
    """間隔を空けて GET。403 は待ってリトライ。"""
    global _last
    for attempt in range(MAX_RETRIES + 1):
        time.sleep(max(0.0, SLEEP - (time.time() - _last)))
        _last = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 403 and attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
                continue
            raise


def _request(params: dict, endpoint: str = "getStatsData") -> dict:
    query = urllib.parse.urlencode({"appId": _app_id(), "lang": "J", **params})
    return json.loads(_get(f"{API}{endpoint}?{query}").decode("utf-8"))


def _as_list(x) -> list:
    return [] if x is None else x if isinstance(x, list) else [x]


def latest_catalog_file(stats_code: str, resource_pattern: str, *,
                        dataset_pattern: str | None = None, exclude_dataset: str | None = None,
                        search_word: str | None = None) -> dict:
    """getDataCatalog で統計の公表ファイルを走査し、名前が一致する最新 (RELEASE_DATE 最大) のファイルを返す。

    DB (getStatsData) に無い/更新が止まっている統計でも、公表 Excel はカタログから取れることが多い。
    statInfId は毎月の公表で変わるため、ID を固定せず毎回ここで探す。
    """
    found, start = [], None
    while True:
        params = {"statsCode": stats_code, "limit": 100}  # limit は 100 以下でないとエラー
        if search_word:
            params["searchWord"] = search_word
        if start:
            params["startPosition"] = start
        body = _request(params, "getDataCatalog")["GET_DATA_CATALOG"]
        inf = body.get("DATA_CATALOG_LIST_INF", {})
        for cat in _as_list(inf.get("DATA_CATALOG_INF")):
            dataset = cat["DATASET"]["TITLE"]["NAME"]
            if exclude_dataset and re.search(exclude_dataset, dataset):
                continue
            if dataset_pattern and not re.search(dataset_pattern, dataset):
                continue
            for res in _as_list(cat.get("RESOURCES", {}).get("RESOURCE")):
                name = res["TITLE"]["NAME"]
                if re.search(resource_pattern, name):
                    found.append({"release_date": res.get("RELEASE_DATE", ""), "dataset": dataset,
                                  "name": name, "url": res["URL"]})
        start = inf.get("RESULT_INF", {}).get("NEXT_KEY")
        if not start:
            break
    if not found:
        raise RuntimeError(f"e-Stat カタログ {stats_code}: '{resource_pattern}' に一致するファイルがありません")
    return max(found, key=lambda f: f["release_date"])


def download(url: str) -> bytes:
    return _get(url)


def get_values(stats_id: str, **filters: str) -> tuple[list[dict], dict]:
    """VALUE の全件 (NEXT_KEY を辿る) と、分類コード→名称の辞書を返す。

    filters は cdTab="3", cdCat01="0001" のように API パラメータ名で指定する。
    """
    values, names, start = [], {}, None
    while True:
        params = {"statsDataId": stats_id, "limit": 100000, **filters}
        if start:
            params["startPosition"] = start
        body = _request(params)["GET_STATS_DATA"]
        status = body["RESULT"]["STATUS"]
        if status != 0:
            raise RuntimeError(f"e-Stat {stats_id}: STATUS={status} {body['RESULT'].get('ERROR_MSG')}")
        data = body["STATISTICAL_DATA"]
        if not names:
            objs = data["CLASS_INF"]["CLASS_OBJ"]
            for obj in objs if isinstance(objs, list) else [objs]:
                classes = obj["CLASS"] if isinstance(obj["CLASS"], list) else [obj["CLASS"]]
                names[obj["@id"]] = {c["@code"]: c["@name"] for c in classes}
        vals = data["DATA_INF"]["VALUE"]
        values += vals if isinstance(vals, list) else [vals]
        start = data["RESULT_INF"].get("NEXT_KEY")
        if not start:
            return values, names


def _to_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")  # "-", "***", "…" 等


def monthly_frame(stats_id: str, by: str | None = None, *, where: dict[str, str] | None = None,
                  **filters: str) -> pd.DataFrame:
    """月次データを DataFrame (月初 Timestamp index) で返す。

    - 年月は @time の表示名「2024年1月」から読む (統計ごとに異なる @time コード形式に依存しない)
    - by="cat01" のように指定すると、その分類コードごとに列を分ける
    - where={"area": "全国"} のように、分類の「名称」で行を絞り込める (コードが分からない場合用)
    """
    values, names = get_values(stats_id, **filters)
    rows: dict[tuple, float] = {}
    for v in values:
        if where and any(names.get(k, {}).get(v.get(f"@{k}")) != label for k, label in where.items()):
            continue
        m = re.search(r"(\d{4})年(\d{1,2})月", names.get("time", {}).get(v["@time"], ""))
        if not m:
            continue  # 年平均・四半期などの行
        date = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        rows[(date, v.get(f"@{by}") if by else "value")] = _to_float(v.get("$"))
    s = pd.Series(rows, dtype=float)
    return s.unstack().sort_index() if len(s) else pd.DataFrame()


def monthly_series(stats_id: str, name: str = "", **kwargs) -> pd.Series:
    df = monthly_frame(stats_id, **kwargs)
    if df.shape[1] != 1:
        raise ValueError(f"{stats_id}: 系列が1本に絞れていません (列: {list(df.columns)[:5]})")
    return df.iloc[:, 0].rename(name or stats_id)
