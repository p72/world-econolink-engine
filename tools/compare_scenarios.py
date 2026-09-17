"""複数のシミュレーション結果を比べる1枚絵を作る。

使い方:
    python tools/compare_scenarios.py 2026-06_real 2026-06_hold 2026-06_cut \
        --labels 利上げ 据え置き 利下げ --out outputs/compare_policy.png

outputs/<run-name>.csv をいくつか読んで、主要系列を重ね描きした PNG を書き出す。
シナリオは最大8本まで（配色のスロット数）。
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

# 配色は dataviz スキルの検証済みパレット（categorical スロット1〜8）を順に使う。
# 先頭3色は scripts/validate_palette.js --pairs all で全項目 PASS（最悪ペアの
# CVD ΔE 9.2、通常視 ΔE 24.0）。ただし aqua はコントラスト 2.74 と 3:1 未満なので、
# 各線の末端に数値を直接添えること（relief rule）と、線種を変えることが条件。
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_STYLES = ["-", "--", "-.", (0, (3, 1, 1, 1, 1, 1)), ":", "-", "--", "-."]
C_ACTUAL = "#9b9a95"  # 実績（全シナリオ共通なので1本だけ引く）
INK, INK_SUB, GRID = "#0b0b0b", "#52514e", "#d8d6d1"

# 既定で描くパネル（列名。--panels で差し替えられる）
LINE_PANELS = [
    "jp_short_rate", "jp_gdp",
    "jp_inflation", "jp_unemployment",
    "jp_gdp_gap", "jp_consumption",
    "jp_long_rate", "jp_usd_jpy",
]

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


def _end_labels(ax, x, values: list[float]) -> None:
    """最終点だけ数値を直接添える（全点に数字を振らない）。
    値が近いとラベルが重なるので、値の順に上から並べ直して間隔を空ける。"""
    lo, hi = ax.get_ylim()
    span = max(hi - lo, 1e-9)
    pts = sorted(((y, c) for y, c in values if pd.notna(y)), key=lambda t: -t[0])
    min_gap = 0.075 * span  # これより近いラベルは押し下げる
    placed: list[float] = []
    for y, _ in pts:
        placed.append(y if not placed else min(y, placed[-1] - min_gap))
    # 押し下げた結果が軸の下にはみ出たら、重なりを保ったまま全体を持ち上げる
    if placed and placed[-1] < lo:
        shift = min(lo - placed[-1], hi - placed[0])
        placed = [p + shift for p in placed]
    # データ座標の差 → ポイント。ax.bbox は px なので 72/dpi でポイントに直す
    #（ここを px のまま渡すと dpi 倍だけ過大にずれる）
    to_points = ax.bbox.height / span * 72 / ax.figure.dpi
    for (y, color), pos in zip(pts, placed):
        fmt = f"{y:,.1f}" if abs(y) >= 100 else f"{y:.2f}"
        ax.annotate(fmt, xy=(x, y), xytext=(5, (pos - y) * to_points),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=color)


def build(runs: list[pd.DataFrame], labels: list[str], note: str, title: str,
          out: Path, panels: list[str]) -> Path:
    _set_japanese_font()
    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#ffffff"

    first = runs[0]
    base = first.index[first["period"] == 0][0]
    fc = first["period"] >= 0      # 基準月＋予測期
    hist = first["period"] <= 0    # 実績期

    nrow = -(-len(panels) // 2)
    # 注記は GridSpec の行に入れると行間（hspace）に押されて下が切れるので、
    # グリッドの外に出して、図の下余白をそのぶん確保してそこに置く。
    # 1行 = 9.2pt × linespacing 1.75 ≒ 0.23 インチ。最下段パネルの目盛りラベル
    # （45度）もこの帯に垂れ下がるので、そのぶん TICK_HANG を別に確保する。
    tick_hang = 0.75
    note_h = 0.23 * (note.count("\n") + 1) + tick_hang + 0.25 if note else 0.0
    # 上余白 1.0 インチ = 図タイトル + 1段目のパネル見出しのぶん
    fig_h = 3.05 * nrow + 1.35 + note_h
    fig = plt.figure(figsize=(13.2, fig_h))
    gs = GridSpec(nrow, 2, figure=fig, hspace=0.62, wspace=0.18)

    for i, colname in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        # 実績部分は全シナリオ共通なのでグレーで1本だけ引く
        ax.plot(first.index[hist], first.loc[hist, colname],
                color=C_ACTUAL, lw=2.0, label="実績", zorder=2)
        ends = []
        for j, (df, label) in enumerate(zip(runs, labels)):
            color = SERIES_COLORS[j % len(SERIES_COLORS)]
            ax.plot(df.index[fc], df.loc[fc, colname], color=color, lw=2.0,
                    ls=SERIES_STYLES[j % len(SERIES_STYLES)], label=label, zorder=3 + j)
            ends.append((df[colname].iloc[-1], color))
        _end_labels(ax, first.index[-1], ends)
        ax.axvline(base, color=INK_SUB, ls=":", lw=1.1, zorder=1)
        ax.set_title(heading_of(colname), fontsize=10.5, color=INK, pad=8, loc="left")
        _style_axis(ax)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_SUB)

    # --- 注記（図の下余白に直接置く） --------------------------------------
    if note:
        fig.text(0.055, (note_h - tick_hang) / fig_h, note, va="top", ha="left",
                 fontsize=9.2, color=INK_SUB, linespacing=1.75)

    fig.suptitle(title, fontsize=15, color=INK, x=0.055, ha="left", y=1 - 0.38 / fig_h)
    fig.subplots_adjust(top=1 - 1.0 / fig_h, bottom=note_h / fig_h, left=0.055, right=0.975)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="複数のシミュレーション結果を比較する PNG を作る")
    p.add_argument("runs", nargs="*", help="outputs/<run-name>.csv の run 名（2本以上）")
    p.add_argument("--labels", nargs="+", default=None, help="凡例の名前（既定は run 名）")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default="日銀の金融政策シナリオ比較")
    p.add_argument("--note-file", default=None, help="注記を書いたテキストファイル")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--panels", nargs="+", default=LINE_PANELS,
                   help=f"描く列名（既定: {' '.join(LINE_PANELS)}）")
    p.add_argument("--list-vars", action="store_true", help="使える列名を一覧して終了")
    args = p.parse_args()

    if args.list_vars:
        for v, lab in ALL_VARS.items():
            print(f"{col(v):32s} {lab}")
        return

    if len(args.runs) < 2:
        p.error("run 名を2つ以上指定してください")
    labels = args.labels or args.runs
    if len(labels) != len(args.runs):
        p.error(f"--labels は run と同じ数だけ必要です（run {len(args.runs)} / labels {len(labels)}）")

    od = Path(args.outputs)
    runs = [pd.read_csv(od / f"{r}.csv", index_col=0, parse_dates=True) for r in args.runs]
    missing = [c for c in args.panels if c not in runs[0].columns]
    if missing:
        p.error(f"CSV に無い列です: {', '.join(missing)}（--list-vars で一覧）")
    note = Path(args.note_file).read_text(encoding="utf-8").rstrip() if args.note_file else ""
    out = Path(args.out) if args.out else od / ("compare_" + "_vs_".join(args.runs) + ".png")
    print("->", build(runs, labels, note, args.title, out, args.panels))


if __name__ == "__main__":
    main()
