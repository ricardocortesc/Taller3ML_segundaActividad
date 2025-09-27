"""
pipeline_pca.py — Orquestación con Prefect que cumple los 5 puntos del taller.
"""
from __future__ import annotations
from prefect import flow
from pathlib import Path

from ..config.settings import PROJECT_ROOT, ASSETS_DIR, N_COMPONENTS, STANDARDIZE, SEED
from .utils import ensure_dirs
from ..tasks.pca_challenge import (
    load_iris_example, load_tabular, prepare_wine_targets,
    run_pca, plot_variance, save_components,
    compare_original_vs_pca, save_results
)

WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

@flow(name="workshop-pca-pipeline", log_prints=True)
def run_pipeline(
    n_components: int = N_COMPONENTS,
    standardize: bool = STANDARDIZE,
    seed: int = SEED
):
    # 0) carpetas
    ensure_dirs(PROJECT_ROOT, ["data", "assets"])

    # ===== Parte 1 — Ejemplo base (IRIS) según Workflow =====
    X_iris, y_iris = load_iris_example()
    pca_out_iris = run_pca(X_iris, n_components=n_components, standardize=standardize, seed=seed)
    plot_variance(pca_out_iris["explained_ratio"], pca_out_iris["explained_cum"])
    save_components(pca_out_iris["pcs_df"], pca_out_iris["explained_ratio"], ASSETS_DIR)

    # ===== Parte 2 — Aplicar workflow al dataset de vino =====
    df_wine = load_tabular(WINE_URL, sep=";")

    # Multiclase
    Xw_mc, yw_mc = prepare_wine_targets(df_wine, multiclass=True)
    res_mc = compare_original_vs_pca(Xw_mc, yw_mc, n_components=2, random_state=seed, is_multiclass=True)
    save_results(res_mc, ASSETS_DIR / "wine_multiclass_results.csv")

    # Binaria (good > 6)
    Xw_bin, yw_bin = prepare_wine_targets(df_wine, multiclass=False, binary_threshold=6)
    res_bin = compare_original_vs_pca(Xw_bin, yw_bin, n_components=2, random_state=seed, is_multiclass=False)
    save_results(res_bin, ASSETS_DIR / "wine_binary_results.csv")

    print("\n>> Artefactos guardados en:", ASSETS_DIR)
