"""World EconoLink Engine CLI.

  python run.py demo                          合成データで 生成→推定→シミュレーション→出力 を一括実行
  python run.py synth                         data/synthetic.csv を生成
  python run.py fetch                         data/us.csv (FRED・BLS) と data/jp.csv (e-Stat・厚労省・日銀・内閣府・財務省, 要 ESTAT_APP_ID) を取得
  python run.py train  --data synthetic.csv   28方程式を推定し models/*.pkl に保存
  python run.py simulate --scenario scenarios/2026-02.json [--data ...] [--run-name ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from econolink.data import US_SOURCES, fetch_us, load_dataset, make_synthetic, write_template
from econolink.engine import Engine, Scenario
from econolink.estimate import load_models, train_all
from econolink.exporter import Exporter
from econolink.sources_jp import SOURCES, fetch_japan

ROOT = Path(__file__).parent
DATA, MODELS, OUTPUTS = ROOT / "data", ROOT / "models", ROOT / "outputs"
DEFAULT_DATA = ["us.csv", "jp.csv", "jp_manual.csv"]  # 後ろほど優先


def cmd_synth(args):
    DATA.mkdir(exist_ok=True)
    make_synthetic(end=args.end, seed=args.seed).to_csv(DATA / "synthetic.csv")
    print(f"-> {DATA / 'synthetic.csv'}")


def cmd_fetch(args):
    DATA.mkdir(exist_ok=True)
    fetch_us().to_csv(DATA / "us.csv")
    print(f"-> {DATA / 'us.csv'}")
    for col_name, desc in US_SOURCES.items():
        print(f"   {col_name:28s} {desc}")
    if not os.environ.get("ESTAT_APP_ID"):
        write_template(DATA / "template_jp_manual.csv")
        print("環境変数 ESTAT_APP_ID が未設定のため日本系列の自動取得をスキップしました。")
        print(f"-> {DATA / 'template_jp_manual.csv'} (手入力する場合は jp_manual.csv として保存)")
        return
    print("日本系列を取得中 (e-Stat / 厚労省 / 日銀 / 内閣府 / 財務省) ...")
    fetch_japan().to_csv(DATA / "jp.csv")
    print(f"-> {DATA / 'jp.csv'}")
    for col_name, desc in SOURCES.items():
        print(f"   {col_name:26s} {desc}")


def _data(args):
    files = args.data or [f for f in DEFAULT_DATA if (DATA / f).exists()]
    return load_dataset(DATA, files)


def cmd_train(args):
    train_all(_data(args), MODELS, train_end=args.train_end, train_start=args.train_start)


def cmd_simulate(args):
    sc = Scenario(**json.loads(Path(args.scenario).read_text(encoding="utf-8")))
    engine = Engine(load_models(MODELS), _data(args))
    result = engine.run(sc)
    csv_path, png_paths = Exporter(OUTPUTS).export(result, args.run_name)
    cols = ["period", "jp_gdp", "jp_gdp_gap", "jp_inflation", "jp_short_rate", "jp_long_rate",
            "jp_usd_jpy", "us_short_rate", "world_oil_price"]
    print(result[[c for c in cols if c in result]].round(2).to_string())
    print(f"-> {csv_path}\n-> {png_paths[0].parent} (PNG {len(png_paths)}枚)")


def cmd_demo(args):
    args.end, args.seed = "2026-02-01", 0
    cmd_synth(args)
    args.data = ["synthetic.csv"]
    args.train_end, args.train_start = "2025-01-01", "2007-01-01"
    cmd_train(args)
    args.scenario, args.run_name = ROOT / "scenarios" / "2026-02.json", "2026-02_output"
    cmd_simulate(args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth"); s.add_argument("--end", default="2026-02-01"); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)
    sub.add_parser("fetch").set_defaults(func=cmd_fetch)
    for name, fn in [("train", cmd_train), ("simulate", cmd_simulate)]:
        s = sub.add_parser(name)
        s.add_argument("--data", nargs="+", help="data/ 内のCSV (後ろほど優先)。省略時は us.csv jp.csv jp_manual.csv")
        if name == "train":
            s.add_argument("--train-start", default="2007-01-01")
            s.add_argument("--train-end", default="2025-01-01")
        else:
            s.add_argument("--scenario", default=str(ROOT / "scenarios" / "2026-02.json"))
            s.add_argument("--run-name", default=None)
        s.set_defaults(func=fn)
    sub.add_parser("demo").set_defaults(func=cmd_demo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
