"""
pipeline_pca.py — Orquestación con Prefect que cumple los 5 puntos del taller.
VERSIÓN COMPLETA: Incluye análisis PCA para dataset de vino.
"""
from __future__ import annotations
from prefect import flow
from pathlib import Path

from src.config.settings import PROJECT_ROOT, ASSETS_DIR, N_COMPONENTS, STANDARDIZE, SEED
from src.tasks.pca_challenge import (
    load_iris_example,
    load_tabular,
    prepare_wine_targets,
    run_pca,
    plot_variance,
    save_components,
    compare_original_vs_pca,
    save_results
)

WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


@flow(name="workshop-pca-pipeline", log_prints=True)
def run_pipeline(
    n_components: int = N_COMPONENTS,
    standardize: bool = STANDARDIZE,
    seed: int = SEED
):
    """
    Pipeline principal del taller PCA.
    
    Cumple los 5 puntos:
    1. Organizar ejemplo IRIS según Data Science Workflow
    2. Aplicar workflow al dataset de vino
    3. Completar pasos del Supervised Learning Workflow
    4. Comparar clasificación con features originales vs PCA
    5. Clasificación binaria (quality > 6) vs multiclase
    """
    
    # ===== SETUP =====
    print("\n" + "="*70)
    print("WORKSHOP PCA - PIPELINE COMPLETO")
    print("="*70 + "\n")
    
    # Crear directorios necesarios
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "iris").mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "wine").mkdir(parents=True, exist_ok=True)
    
    # ===================================================================
    # PUNTO 1: EJEMPLO IRIS - Data Science Workflow
    # ===================================================================
    print("\n" + "="*70)
    print("PUNTO 1: EJEMPLO IRIS - Data Science Workflow")
    print("="*70 + "\n")
    
    # 1.1 Cargar datos
    X_iris, y_iris = load_iris_example()
    
    # 1.2 Aplicar PCA
    pca_out_iris = run_pca(
        X_iris,
        n_components=n_components,
        standardize=standardize,
        seed=seed
    )
    
    # 1.3 Visualizar varianza explicada
    plot_variance(
        pca_out_iris["explained_ratio"],
        pca_out_iris["explained_cum"]
    )
    
    # 1.4 Guardar componentes
    save_components(
        pca_out_iris["pcs_df"],
        pca_out_iris["explained_ratio"],
        ASSETS_DIR / "iris"
    )
    
    # ===================================================================
    # PUNTO 2: DATASET VINO - Aplicar Workflow
    # ===================================================================
    print("\n" + "="*70)
    print("PUNTO 2: DATASET VINO - Supervised Learning Workflow")
    print("="*70 + "\n")
    
    # 2.1 Cargar dataset
    df_wine = load_tabular(WINE_URL, sep=";")
    
    # 2.2 Preparar features (sin target aún) para análisis PCA
    X_wine_all = df_wine.drop(columns=['quality']).copy()
    y_wine_quality = df_wine['quality'].copy()
    
    # 2.3 Aplicar PCA al dataset completo de vino (análisis exploratorio)
    print("\nAnálisis PCA del Dataset de Vino:")
    pca_out_wine = run_pca(
        X_wine_all,
        n_components=n_components,
        standardize=standardize,
        seed=seed
    )
    
    # 2.4 Visualizar varianza explicada
    plot_variance(
        pca_out_wine["explained_ratio"],
        pca_out_wine["explained_cum"]
    )
    
    # 2.5 Guardar componentes del vino
    save_components(
        pca_out_wine["pcs_df"],
        pca_out_wine["explained_ratio"],
        ASSETS_DIR / "wine"
    )
    
    print(f"\nResumen PCA Wine:")
    print(f"PC1: {pca_out_wine['explained_ratio'][0]:.2%} varianza")
    print(f"PC2: {pca_out_wine['explained_ratio'][1]:.2%} varianza")
    print(f"Total (2 PCs): {pca_out_wine['explained_cum'][1]:.2%} varianza")
    
    # ===================================================================
    # PUNTO 3 & 4: CLASIFICACIÓN MULTICLASE + COMPARACIÓN
    # ===================================================================
    print("\n" + "="*70)
    print("PUNTO 3 & 4: CLASIFICACIÓN MULTICLASE")
    print("         (poor ≤5, fair=6, good ≥7)")
    print("="*70 + "\n")
    
    # 3.1 Preparar datos
    X_wine_mc, y_wine_mc = prepare_wine_targets(df_wine, multiclass=True)
    
    # 3.2 Comparar: Features originales vs PCA (PUNTO 4)
    results_mc = compare_original_vs_pca(
        X_wine_mc,
        y_wine_mc,
        n_components=n_components,
        random_state=seed,
        is_multiclass=True
    )
    
    # 3.3 Guardar resultados
    save_results(results_mc, ASSETS_DIR / "wine_multiclass_results.csv")
    
    # ===================================================================
    # PUNTO 5: CLASIFICACIÓN BINARIA + COMPARACIÓN
    # ===================================================================
    print("\n" + "="*70)
    print("PUNTO 5: CLASIFICACIÓN BINARIA")
    print("         (quality > 6 = 'good')")
    print("="*70 + "\n")
    
    # 5.1 Preparar datos
    X_wine_bin, y_wine_bin = prepare_wine_targets(
        df_wine,
        multiclass=False,
        binary_threshold=6
    )
    
    # 5.2 Comparar: Features originales vs PCA
    results_bin = compare_original_vs_pca(
        X_wine_bin,
        y_wine_bin,
        n_components=n_components,
        random_state=seed,
        is_multiclass=False
    )
    
    # 5.3 Guardar resultados
    save_results(results_bin, ASSETS_DIR / "wine_binary_results.csv")
    
    # ===================================================================
    # RESUMEN FINAL
    # ===================================================================
    print("\n" + "="*70)
    print("Resultados Finales")
    print("="*70)
    
    print("\nIRIS:")
    print(f"Varianza explicada (2 PCs): {pca_out_iris['explained_cum'][1]:.2%}")
    
    print("\nWINE:")
    print(f"Varianza explicada (2 PCs): {pca_out_wine['explained_cum'][1]:.2%}")
    
    print("\nCLASIFICACIÓN MULTICLASE:")
    print(f"Baseline (todas las features): {results_mc['baseline_accuracy']:.4f}")
    print(f"PCA ({n_components} componentes):       {results_mc['pca_accuracy']:.4f}")
    print(f"Diferencia:                     {results_mc['accuracy_diff']:+.4f}")
    print(f"Varianza explicada:             {results_mc['variance_explained']:.2%}")
    
    print("\nCLASIFICACIÓN BINARIA:")
    print(f"Baseline (todas las features): {results_bin['baseline_accuracy']:.4f}")
    print(f"PCA ({n_components} componentes):       {results_bin['pca_accuracy']:.4f}")
    print(f"Diferencia:                     {results_bin['accuracy_diff']:+.4f}")
    print(f"Varianza explicada:             {results_bin['variance_explained']:.2%}")
    
    print("\nMULTICLASE vs BINARIA:")
    mejor = "BINARIA" if results_bin['baseline_accuracy'] > results_mc['baseline_accuracy'] else "MULTICLASE"
    print(f"Mejor clasificación: {mejor}")
    
    print("\n" + "="*70)
    print("Fin")
    print("="*70)
    print(f"\nArtefactos en: {ASSETS_DIR}")
    print("\nArchivos generados:")
    print("iris/components.csv")
    print("iris/explained_ratio.csv")
    print("wine/components.csv")
    print("wine/explained_ratio.csv")
    print("wine_multiclass_results.csv")
    print("wine_binary_results.csv")
    print("\n" + "="*70 + "\n")