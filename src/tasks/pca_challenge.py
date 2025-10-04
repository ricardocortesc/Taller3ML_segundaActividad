"""
pca_challenge.py — Funciones/Tareas para PCA del taller.
Incluye todas las funciones necesarias para el pipeline completo.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from prefect import task, get_run_logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# =========================
#        CARGA DE DATOS
# =========================

@task(name="load_iris_example", log_prints=True)
def load_iris_example() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga el dataset Iris como DataFrame.
    
    Returns
    -------
    X : DataFrame con features
    y : Series con target
    """
    logger = get_run_logger()
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()
    logger.info(f"Dataset IRIS cargado | X shape={X.shape} | y shape={y.shape}")
    return X, y


@task(name="load_tabular", log_prints=True)
def load_tabular(url_or_path: str, sep: str = ",") -> pd.DataFrame:
    """
    Carga un dataset tabular desde URL o archivo local.
    
    Parameters
    ----------
    url_or_path : URL o ruta al archivo CSV
    sep : separador (por defecto ',')
    
    Returns
    -------
    DataFrame con los datos
    """
    logger = get_run_logger()
    df = pd.read_csv(url_or_path, sep=sep)
    logger.info(f"Dataset cargado desde {url_or_path} | shape={df.shape}")
    return df


@task(name="prepare_wine_targets", log_prints=True)
def prepare_wine_targets(
    df: pd.DataFrame,
    multiclass: bool = True,
    binary_threshold: int = 6
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara features y targets para el dataset de vino.
    
    Parameters
    ----------
    df : DataFrame con el dataset de vino
    multiclass : Si True, crea 3 clases (poor/fair/good). Si False, binario (good/not_good)
    binary_threshold : umbral para clasificación binaria (quality > threshold = 'good')
    
    Returns
    -------
    X : DataFrame con features (sin columna quality)
    y : Series con targets
    """
    logger = get_run_logger()
    
    # Separar features y target
    X = df.drop(columns=['quality']).copy()
    
    if multiclass:
        # Clasificación multiclase: poor (≤5), fair (6), good (≥7)
        def quality_to_class(quality):
            if quality <= 5:
                return 'poor'
            elif quality <= 6:
                return 'fair'
            else:
                return 'good'
        
        y = df['quality'].apply(quality_to_class)
        logger.info(f"Targets multiclase creados | Distribución:\n{y.value_counts()}")
    else:
        # Clasificación binaria: quality > threshold
        y = (df['quality'] > binary_threshold).astype(int)
        logger.info(f"Targets binarios creados (quality > {binary_threshold}) | Distribución:\n{y.value_counts()}")
    
    return X, y


# =========================
#        PCA
# =========================

@task(name="run_pca", log_prints=True)
def run_pca(
    X: pd.DataFrame,
    n_components: int = 2,
    standardize: bool = True,
    seed: int = 42
) -> Dict[str, object]:
    """
    Aplica StandardScaler (opcional) y PCA a X.
    
    Returns
    -------
    dict con:
      - 'pcs_df'            : DataFrame con componentes principales
      - 'explained_ratio'   : varianza explicada por componente
      - 'explained_cum'     : varianza acumulada
      - 'pca'               : objeto PCA entrenado
      - 'scaler'            : StandardScaler | None
      - 'feature_names'     : nombres de features originales
    """
    logger = get_run_logger()
    feature_names = list(X.columns)
    
    scaler = None
    X_mat = X.values.astype(float)
    
    if standardize:
        scaler = StandardScaler()
        X_mat = scaler.fit_transform(X_mat)
        logger.info("Datos estandarizados (StandardScaler)")
    
    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(X_mat)
    
    explained_ratio = pca.explained_variance_ratio_
    explained_cum = np.cumsum(explained_ratio)
    pcs_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pcs_df = pd.DataFrame(pcs, columns=pcs_cols, index=X.index)
    
    logger.info(f"PCA completado: n_components={n_components}")
    logger.info(f"   Varianza explicada: {np.round(explained_ratio, 4)}")
    logger.info(f"   Varianza acumulada: {np.round(explained_cum, 4)}")
    
    return {
        "pcs_df": pcs_df,
        "explained_ratio": explained_ratio,
        "explained_cum": explained_cum,
        "pca": pca,
        "scaler": scaler,
        "feature_names": feature_names,
    }


@task(name="plot_variance", log_prints=True)
def plot_variance(explained_ratio: np.ndarray, explained_cum: np.ndarray) -> None:
    """
    Grafica la varianza explicada y acumulada.
    """
    if explained_ratio is None or len(explained_ratio) == 0:
        print("explained_ratio vacío; no hay nada para graficar.")
        return
    
    x = np.arange(1, len(explained_ratio) + 1)
    
    # Varianza explicada por componente
    plt.figure(figsize=(10, 4))
    plt.bar(x, explained_ratio, alpha=0.8, color='steelblue')
    plt.xlabel("Componente Principal")
    plt.ylabel("Varianza explicada")
    plt.title("Varianza explicada por componente")
    plt.xticks(x)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Varianza acumulada
    if explained_cum is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(x, explained_cum, marker="o", linewidth=2, markersize=8, color='darkgreen')
        plt.xlabel("Componente Principal")
        plt.ylabel("Varianza explicada acumulada")
        plt.title("Varianza explicada acumulada")
        plt.ylim(0, 1.05)
        plt.xticks(x)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


@task(name="save_components", log_prints=True)
def save_components(
    pcs_df: pd.DataFrame,
    explained_ratio: np.ndarray,
    out_dir: Path
) -> None:
    """
    Guarda componentes principales y varianza explicada a CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pcs_path = out_dir / "components.csv"
    var_path = out_dir / "explained_ratio.csv"
    
    pcs_df.to_csv(pcs_path, index=True)
    pd.DataFrame({"explained_ratio": explained_ratio}).to_csv(var_path, index=False)
    print(f"Guardado: {pcs_path}")
    print(f"Guardado: {var_path}")


# =========================
#   CLASIFICACIÓN Y COMPARACIÓN
# =========================

@task(name="compare_original_vs_pca", log_prints=True)
def compare_original_vs_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int = 2,
    random_state: int = 42,
    is_multiclass: bool = True
) -> Dict[str, float]:
    """
    Compara el rendimiento de clasificación usando:
    1. Todas las features originales (baseline)
    2. Solo n_components de PCA
    
    Returns
    -------
    dict con métricas de comparación
    """
    logger = get_run_logger()
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # ===== BASELINE: Todas las features originales =====
    logger.info("Entrenando modelo BASELINE (todas las features)...")
    scaler_baseline = StandardScaler()
    X_train_scaled = scaler_baseline.fit_transform(X_train)
    X_test_scaled = scaler_baseline.transform(X_test)
    
    knn_baseline = KNeighborsClassifier(n_neighbors=5)
    knn_baseline.fit(X_train_scaled, y_train)
    y_pred_baseline = knn_baseline.predict(X_test_scaled)
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    
    logger.info(f"Accuracy (baseline): {baseline_accuracy:.4f}")
    print("\nClassification Report (BASELINE):")
    print(classification_report(y_test, y_pred_baseline))
    
    # ===== PCA: Solo n_components =====
    logger.info(f"Entrenando modelo PCA ({n_components} componentes)...")
    scaler_pca = StandardScaler()
    X_train_scaled_pca = scaler_pca.fit_transform(X_train)
    X_test_scaled_pca = scaler_pca.transform(X_test)
    
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled_pca)
    X_test_pca = pca.transform(X_test_scaled_pca)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    
    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_pca = knn_pca.predict(X_test_pca)
    pca_accuracy = accuracy_score(y_test, y_pred_pca)
    
    logger.info(f"Accuracy (PCA): {pca_accuracy:.4f}")
    logger.info(f"Varianza explicada: {variance_explained:.2%}")
    print("\nClassification Report (PCA):")
    print(classification_report(y_test, y_pred_pca))
    
    # ===== COMPARACIÓN =====
    accuracy_diff = pca_accuracy - baseline_accuracy
    logger.info(f"\nDiferencia (PCA - Baseline): {accuracy_diff:+.4f}")
    
    return {
        "baseline_accuracy": baseline_accuracy,
        "pca_accuracy": pca_accuracy,
        "accuracy_diff": accuracy_diff,
        "variance_explained": variance_explained,
        "n_components": n_components,
        "n_features_original": X.shape[1]
    }


@task(name="save_results", log_prints=True)
def save_results(results: Dict, output_path: Path) -> None:
    """
    Guarda los resultados de la comparación a CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame([results])
    df_results.to_csv(output_path, index=False)
    print(f"Resultados guardados en: {output_path}")