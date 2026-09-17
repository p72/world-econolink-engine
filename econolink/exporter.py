"""出力 (論文 第9章): 月次時系列 CSV と、8パネル/枚のダッシュボード PNG。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .variables import ALL_VARS, col

BAR_VARS = {"jp.gdp_gap", "us.gdp_gap"}  # 符号に意味のある系列は棒グラフ


def _set_japanese_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            return


class Exporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(self, result, run_name: str | None = None) -> tuple[Path, list[Path]]:
        """<run_name>.csv と <run_name>_charts/dashboard_NN.png を書き出す。"""
        name = run_name or "macro_indicators"
        csv_path = self.out_dir / f"{name}.csv"
        result.to_csv(csv_path, encoding="utf-8-sig", float_format="%.4f")
        chart_dir = self.out_dir / f"{name}_charts"
        chart_dir.mkdir(exist_ok=True)
        for old in chart_dir.glob("dashboard_*.png"):  # パネル数が減った場合に古い画像を残さない
            old.unlink()
        return csv_path, self._png(result, chart_dir)

    def _png(self, df, chart_dir: Path, per_page: int = 8, dpi: int = 110) -> list[Path]:
        _set_japanese_font()
        vars_ = [v for v in ALL_VARS if col(v) in df and df[col(v)].notna().any()]
        actual = df["period"] <= 0
        forecast = df["period"] >= 0
        base = df.index[df["period"] == 0][0]
        x = df.index
        paths = []
        for page, start in enumerate(range(0, len(vars_), per_page), 1):
            chunk = vars_[start:start + per_page]
            nrows = -(-len(chunk) // 2)  # 画像の高さはパネル数に合わせる (余白を作らない)
            fig, axes = plt.subplots(nrows, 2, figsize=(11.7, 4.1 * nrows), squeeze=False)
            for ax, var in zip(axes.flat, chunk):
                s = df[col(var)]
                if var in BAR_VARS:
                    for xi, yi, is_actual in zip(x, s, actual):
                        ax.bar(xi, yi, width=20, color="#2a78c8" if yi >= 0 else "#d9534f",
                               alpha=0.45 if is_actual else 1.0)
                    ax.axhline(0, color="#555", lw=0.8)
                else:
                    ax.plot(x[actual], s[actual], color="#888", lw=1.8, label="実績")
                    ax.plot(x[forecast], s[forecast], color="#2a78c8", lw=1.8, ls="--", label="予測")
                ax.axvline(base, color="#333", ls=":", lw=1)
                ax.set_title(f"{ALL_VARS[var]}  ― {col(var)}", fontsize=10)
                ax.grid(alpha=0.3)
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            for ax in list(axes.flat)[len(chunk):]:
                ax.axis("off")
            fig.tight_layout()
            path = chart_dir / f"dashboard_{page:02d}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            paths.append(path)
        return paths
