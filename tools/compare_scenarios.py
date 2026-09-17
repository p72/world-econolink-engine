"""2つのシミュレーション結果を比べる1枚絵を作る。

使い方:
    python tools/compare_scenarios.py 2026-06_real 2026-06_hold \
        --labels 利上げ 据え置き --out outputs/compare_hike_vs_hold.png

outputs/<run-name>.csv を2つ読んで、主要系列を重ね描きしたパネルと、
差分（1つ目 − 2つ目）のパネルを並べた PNG を書き出す。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from econolink.exporter import _set_japanese_font  # noqa: E402
from econolink.variables import ALL_VARS, col  # noqa: E402

# 配色は dataviz スキルの検証済みパレット。2系列は categorical スロット1・2、
# 差分パネルの符号は diverging の blue↔red を使う（validate_palette.js で全項目 PASS）。
C_A, C_B = "#2a78d6", "#eb6834"      # シナリオ A / B
C_POS, C_NEG = "#2a78d6", "#e34948"  # 差分の +/−
INK, INK_SUB, GRID = "#0b0b0b", "#52514e", "#d8d6d1"

# 既定で重ね描きするパネル（列名。--panels で差し替えられる）
LINE_PANELS = [
    "jp_short_rate", "jp_gdp",
    "jp_inflation", "jp_unemployment",
    "jp_gdp_gap", "jp_consumption",
    "jp_long_rate", "jp_usd_jpy",
]
# 既定で差分を棒にするパネル（列名。--diff-panels で差し替えられる）
DIFF_PANELS = ["jp_gdp", "jp_unemployment", "jp_inflation", "jp_gdp_gap"]

# シナリオで外から与える変数。見出しに内生でないことを書き添える。
EXOGENOUS = {"jp_short_rate", "world_oil_price"}
COUNTRY = {"jp_": "日本：", "us_": "米国：", "world_": ""}


def heading_of(colname: str) -> str:
    """列名から見出しを作る。ラベルは variables.ALL_VARS の定義をそのまま使う。
    日米で同じラベル（実質GDP 前年比 など）があるので、国名を前に付けて区別する。"""
    label = {col(v): lab for v, lab in ALL_VARS.items()}.get(colname, colname)
    prefix = next((p for k, p in COUNTRY.items() if colname.startswith(k)), "")
    suffix = "　← シナリオで外から与える" if colname in EXOGENOUS else ""
    return f"{prefix}{label}{suffix}"


def _style_axis(ax) -> None:
    ax.grid(alpha=0.35, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SUB, labelsize=8.5, length=0)
    ax.tick_params(axis="x", labelrotation=45)


def _end_labels(ax, x, ya, yb, ca, cb) -> None:
    """最終点だけ数値を直接添える（全点に数字を振らない）。
    2本の値が近いとラベルが重なるので、そのときだけ上下にずらす。"""
    lo, hi = ax.get_ylim()
    gap = 0 if pd.isna(ya) or pd.isna(yb) else abs(ya - yb) / max(hi - lo, 1e-9)
    dy = 0 if gap > 0.10 else 9  # 軸レンジの10%より近ければ振り分ける
    # 値が大きいほうを上にずらす（上下を取り違えると線とラベルの対応が読めなくなる）
    sign_a = 1 if pd.isna(yb) or ya >= yb else -1
    for y, color, sign in ((ya, ca, sign_a), (yb, cb, -sign_a)):
        if pd.isna(y):
            continue
        fmt = f"{y:,.1f}" if abs(y) >= 100 else f"{y:.2f}"
        ax.annotate(fmt, xy=(x, y), xytext=(5, sign * dy), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=color)


def build(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str,
          note: str, title: str, out: Path,
          panels: list[str], diff_panels: list[str]) -> Path:
    _set_japanese_font()
    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#ffffff"

    base = a.index[a["period"] == 0][0]
    fc = a["period"] >= 0          # 基準月＋予測期
    hist = a["period"] <= 0        # 実績期

    line_rows = -(-len(panels) // 2)
    diff_rows = -(-len(diff_panels) // 2)
    nrow = line_rows + diff_rows
    # 注記は GridSpec の行に入れると行間（hspace）に押されて下が切れるので、
    # グリッドの外に出して、図の下余白をそのぶん確保してそこに置く。
    # 1行 = 9.2pt × linespacing 1.75 ≒ 0.23 インチ。最下段パネルの
    # 目盛りラベル（45度）がこの帯に垂れ下がるので、余白は 1.05 と多めに取る。
    note_h = 0.23 * (note.count("\n") + 1) + 1.05 if note else 0.0
    # 上余白 1.0 インチ = 図タイトル + 1段目のパネル見出しのぶん
    fig_h = 3.05 * nrow + 1.35 + note_h
    fig = plt.figure(figsize=(13.2, fig_h))
    gs = GridSpec(nrow, 2, figure=fig, hspace=0.62, wspace=0.18)

    # --- 水準の重ね描き ----------------------------------------------------
    for i, colname in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        # 実績部分は共通なのでグレーで1本だけ引く
        ax.plot(a.index[hist], a.loc[hist, colname], color="#9b9a95", lw=2.0, label="実績", zorder=2)
        ax.plot(a.index[fc], a.loc[fc, colname], color=C_A, lw=2.0, label=label_a, zorder=4)
        ax.plot(b.index[fc], b.loc[fc, colname], color=C_B, lw=2.0, ls="--", label=label_b, zorder=3)
        _end_labels(ax, a.index[-1], a[colname].iloc[-1], b[colname].iloc[-1], C_A, C_B)
        ax.axvline(base, color=INK_SUB, ls=":", lw=1.1, zorder=1)
        ax.set_title(heading_of(colname), fontsize=10.5, color=INK, pad=8, loc="left")
        _style_axis(ax)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_SUB)

    # --- 差分（A − B） -----------------------------------------------------
    for i, colname in enumerate(diff_panels):
        ax = fig.add_subplot(gs[line_rows + i // 2, i % 2])
        d = (a[colname] - b[colname])[fc]
        ax.bar(d.index, d.values, width=20,
               color=[C_POS if v >= 0 else C_NEG for v in d.values], zorder=3)
        ax.axhline(0, color=INK_SUB, lw=1.0, zorder=2)
        ax.set_title(f"{label_a}の効果：{heading_of(colname)}の差", fontsize=10.5,
                     color=INK, pad=8, loc="left")
        _style_axis(ax)
        last = d.iloc[-1]
        ax.annotate(f"{last:+.2f}", xy=(d.index[-1], last),
                    xytext=(0, 9 if last >= 0 else -16), textcoords="offset points",
                    ha="center", fontsize=9,
                    color=C_POS if last >= 0 else C_NEG)

    # --- 注記（図の下余白に直接置く） --------------------------------------
    if note:
        fig.text(0.055, (note_h - 0.30) / fig_h, note, va="top", ha="left",
                 fontsize=9.2, color=INK_SUB, linespacing=1.75)

    fig.suptitle(title, fontsize=15, color=INK, x=0.055, ha="left", y=1 - 0.38 / fig_h)
    fig.subplots_adjust(top=1 - 1.0 / fig_h, bottom=note_h / fig_h, left=0.055, right=0.975)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="2つのシミュレーション結果を比較する PNG を作る")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--labels", nargs=2, default=None, help="凡例の名前（既定は run 名）")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default="日銀の利上げシナリオ比較")
    p.add_argument("--note-file", default=None, help="注記を書いたテキストファイル")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--panels", nargs="+", default=LINE_PANELS,
                   help=f"重ね描きする列名（既定: {' '.join(LINE_PANELS)}）")
    p.add_argument("--diff-panels", nargs="+", default=DIFF_PANELS,
                   help=f"差分を棒にする列名（既定: {' '.join(DIFF_PANELS)}）")
    p.add_argument("--list-vars", action="store_true", help="使える列名を一覧して終了")
    args = p.parse_args()

    if args.list_vars:
        for v, lab in ALL_VARS.items():
            print(f"{col(v):32s} {lab}")
        return

    od = Path(args.outputs)
    a = pd.read_csv(od / f"{args.run_a}.csv", index_col=0, parse_dates=True)
    b = pd.read_csv(od / f"{args.run_b}.csv", index_col=0, parse_dates=True)
    missing = [c for c in args.panels + args.diff_panels if c not in a.columns]
    if missing:
        p.error(f"CSV に無い列です: {', '.join(missing)}（--list-vars で一覧）")
    la, lb = args.labels or (args.run_a, args.run_b)
    note = Path(args.note_file).read_text(encoding="utf-8").rstrip() if args.note_file else ""
    out = Path(args.out) if args.out else od / f"compare_{args.run_a}_vs_{args.run_b}.png"
    print("->", build(a, b, la, lb, note, args.title, out, args.panels, args.diff_panels))


if __name__ == "__main__":
    main()
