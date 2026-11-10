"""Estilo común de las gráficas (EDA y evaluación) y utilidades de formato en español."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from src import config  # noqa: E402

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Dos series comparadas (p. ej. entrenamiento/validación o BiLSTM/línea base).
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
# Rampa secuencial azul (de claro a oscuro) para magnitudes como la matriz de confusión.
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.labelcolor": INK_2, "font.size": 10,
})


def fmt_int(v):
    """Entero con separador de miles en formato español (121.187)."""
    return f"{int(v):,}".replace(",", ".")


def fmt_pct(v, decimals=1):
    return f"{v * 100:.{decimals}f} %".replace(".", ",")


def fmt_dec(v, decimals=3):
    return f"{v:.{decimals}f}".replace(".", ",")


THOUSANDS = matplotlib.ticker.FuncFormatter(lambda v, _: fmt_int(v))
PERCENT = matplotlib.ticker.FuncFormatter(lambda v, _: fmt_pct(v, 0))
DECIMAL = matplotlib.ticker.FuncFormatter(lambda v, _: fmt_dec(v, 2))


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figura:", path.relative_to(config.ROOT_DIR))
    return path
