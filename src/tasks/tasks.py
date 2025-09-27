# Package init
"""
pca_challenge.py — Funciones/Tareas para PCA.
- load_data: carga iris (por defecto) o un CSV propio.
- run_pca: aplica (opcional) StandardScaler y PCA.
- plot_variance: grafica varianza explicada.
- save_components: guarda componentes y varianza a CSVs.
- reconstruct_sample: ejemplo de reconstrucción desde PCs.
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
import matplotlib.pyplot as plt


# =========================
#        DATA
# =========================

@task(name="load_data", log_prints=True)
def load_data(
    csv_path: Optional[Path] = None,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Carga un DataFrame:
      - Si csv_path es None: retorna el dataset Iris (features + target).
      - Si csv_path apunta a un CSV: lee, separa target si se indica.

    Returns
    -------
    X : DataFrame (features)
    y : Series | None (target si target_col existe o iris.target)
    """
    logger = get_run_logger()

    if csv_path is None:
        iris = load_iris(as_frame=True)
        X = iris.data.copy()
        y = iris.target.copy()
        logger.info(f"Dataset: IRIS | X shape={X.shape} | y shape={y.shape}")
        return X, y

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"CSV leído: {csv_path} | shape={df.shape}")

    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        logger.info(f"Se separó target '{target_col}'. X shape={X.shape} | y shape={y.shape}")
    else:
        X, y = df, None
        logger.info("No se separó target (no provisto o no existe en columnas).")

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
    Aplica (opcional) StandardScaler y PCA a X.

    Returns
    -------
    dict con:
      - 'pcs_df'            : DataFrame con componentes principales
      - 'explained_ratio'   : np.ndarray con varianza explicada por componente
      - 'explained_cum'     : np.ndarray con varianza acumulada
      - 'pca'               : objeto PCA entrenado
      - 'scaler'            : StandardScaler | None
      - 'feature_names'     : list[str]
    """
    logger = get_run_logger()
    feature_names = list(X.columns)

    scaler = None
    X_mat = X.values.astype(float)

    if standardize:
        scaler = StandardScaler()
        X_mat = scaler.fit_transform(X_mat)
        logger.info("Datos estandarizados (StandardScaler).")

    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(X_mat)

    explained_ratio = pca.explained_variance_ratio_
    explained_cum = np.cumsum(explained_ratio)
    pcs_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pcs_df = pd.DataFrame(pcs, columns=pcs_cols, index=X.index)

    logger.info(f"PCA fit: n_components={n_components}")
    logger.info(f"Varianza explicada: {np.round(explained_ratio, 4)}")
    logger.info(f"Varianza acumulada: {np.round(explained_cum, 4)}")

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

    # Barras de varianza explicada por componente
    plt.figure(figsize=(8, 3.8))
    plt.bar(x, explained_ratio)
    plt.xlabel("Componente Principal")
    plt.ylabel("Varianza explicada")
    plt.title("Varianza explicada por componente")
    plt.xticks(x)
    plt.tight_layout()
    plt.show()

    # Línea de varianza acumulada
    if explained_cum is not None:
        plt.figure(figsize=(8, 3.8))
        plt.plot(x, explained_cum, marker="o")
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
    Guarda:
      - components.csv: PCs del dataset
      - explained_ratio.csv: vector de varianza explicada
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pcs_path = out_dir / "components.csv"
    var_path = out_dir / "explained_ratio.csv"

    pcs_df.to_csv(pcs_path, index=True)
    pd.DataFrame({"explained_ratio": explained_ratio}).to_csv(var_path, index=False)
    print(f"Guardado: {pcs_path}")
    print(f"Guardado: {var_path}")


@task(name="reconstruct_sample", log_prints=True)
def reconstruct_sample(
    original_row: np.ndarray,
    pca: PCA,
    scaler: Optional[StandardScaler]
) -> np.ndarray:
    """
    Reconstruye (aprox) una fila original desde PCs (útil para demos/explicación).
    """
    x = original_row.reshape(1, -1).astype(float)
    if scaler is not None:
        x = scaler.transform(x)

    z = pca.transform(x)
    x_hat = pca.inverse_transform(z)

    if scaler is not None:
        x_hat = scaler.inverse_transform(x_hat)

    return x_hat.ravel()
