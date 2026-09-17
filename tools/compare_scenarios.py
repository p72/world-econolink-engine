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

# 配色は dataviz スキルの検証済みパレット。2系列は categorical スロット1・2、
# 差分パネルの符号は diverging の blue↔red を使う（validate_palette.js で全項目 PASS）。
C_A, C_B = "#2a78d6", "#eb6834"      # シナリオ A / B
C_POS, C_NEG = "#2a78d6", "#e34948"  # 差分の +/−
INK, INK_SUB, GRID = "#0b0b0b", "#52514e", "#d8d6d1"

# 重ね描きするパネル (列名, 見出し)
LINE_PANELS = [
    ("jp_short_rate", "政策金利(%)  ← シナリオで外から与える"),
    ("jp_long_rate", "長期金利(%)"),
    ("jp_inflation", "CPI 前年比(%)"),
    ("jp_gdp_gap", "GDPギャップ(%)"),
    ("jp_usd_jpy", "USD/JPY（上昇＝円安）"),
    ("jp_consumption", "実質消費支出(兆円)"),
]
# 差分を棒で見せるパネル (列名, 見出し)
DIFF_PANELS = [
    ("jp_inflation", "CPI 前年比の差 (pt)"),
    ("jp_gdp_gap", "GDPギャップの差 (pt)"),
]


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
          note: str, title: str, out: Path) -> Path:
    _set_japanese_font()
    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#ffffff"

    base = a.index[a["period"] == 0][0]
    fc = a["period"] >= 0          # 基準月＋予測期
    hist = a["period"] <= 0        # 実績期

    nrow = 4
    # 最終行は注記。行数が多いので、パネル1行分より高めに取る。
    fig = plt.figure(figsize=(13.2, 3.05 * nrow + 3.9))
    gs = GridSpec(nrow + 1, 2, figure=fig,
                  height_ratios=[1] * nrow + [1.35], hspace=0.62, wspace=0.18)

    # --- 水準の重ね描き ----------------------------------------------------
    for i, (colname, heading) in enumerate(LINE_PANELS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        # 実績部分は共通なのでグレーで1本だけ引く
        ax.plot(a.index[hist], a.loc[hist, colname], color="#9b9a95", lw=2.0, label="実績", zorder=2)
        ax.plot(a.index[fc], a.loc[fc, colname], color=C_A, lw=2.0, label=label_a, zorder=4)
        ax.plot(b.index[fc], b.loc[fc, colname], color=C_B, lw=2.0, ls="--", label=label_b, zorder=3)
        _end_labels(ax, a.index[-1], a[colname].iloc[-1], b[colname].iloc[-1], C_A, C_B)
        ax.axvline(base, color=INK_SUB, ls=":", lw=1.1, zorder=1)
        ax.set_title(heading, fontsize=10.5, color=INK, pad=8, loc="left")
        _style_axis(ax)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_SUB)

    # --- 差分（A − B） -----------------------------------------------------
    for i, (colname, heading) in enumerate(DIFF_PANELS):
        ax = fig.add_subplot(gs[3, i])
        d = (a[colname] - b[colname])[fc]
        ax.bar(d.index, d.values, width=20,
               color=[C_POS if v >= 0 else C_NEG for v in d.values], zorder=3)
        ax.axhline(0, color=INK_SUB, lw=1.0, zorder=2)
        ax.set_title(f"{label_a}の効果：{heading}", fontsize=10.5, color=INK, pad=8, loc="left")
        _style_axis(ax)
        last = d.iloc[-1]
        ax.annotate(f"{last:+.2f}", xy=(d.index[-1], last),
                    xytext=(0, 9 if last >= 0 else -16), textcoords="offset points",
                    ha="center", fontsize=9,
                    color=C_POS if last >= 0 else C_NEG)

    # --- 注記 --------------------------------------------------------------
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    ax.text(0, 1, note, va="top", ha="left", fontsize=9.2, color=INK_SUB,
            linespacing=1.75, transform=ax.transAxes)

    fig.suptitle(title, fontsize=15, color=INK, x=0.055, ha="left", y=0.988)
    fig.subplots_adjust(top=0.945, bottom=0.035, left=0.055, right=0.975)
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
    args = p.parse_args()

    od = Path(args.outputs)
    a = pd.read_csv(od / f"{args.run_a}.csv", index_col=0, parse_dates=True)
    b = pd.read_csv(od / f"{args.run_b}.csv", index_col=0, parse_dates=True)
    la, lb = args.labels or (args.run_a, args.run_b)
    note = Path(args.note_file).read_text(encoding="utf-8").rstrip() if args.note_file else ""
    out = Path(args.out) if args.out else od / f"compare_{args.run_a}_vs_{args.run_b}.png"
    print("->", build(a, b, la, lb, note, args.title, out))


if __name__ == "__main__":
    main()
