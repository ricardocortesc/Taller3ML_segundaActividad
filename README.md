# Workshop PCA

_Autogenerado el 2025-09-27 16:44:37_

## Estructura
```
Workshop_PCA/
├─ data/
├─ assets/
├─ src/
│  ├─ config/
│  │  └─ settings.py
│  ├─ flow/
│  │  ├─ pipeline_pca.py
│  │  └─ utils.py
│  └─ tasks/
│     └─ pca_challenge.py   # código extraído del ipynb
├─ main.py
└─ requirements.txt
```

## Uso
```bash
cd Workshop_PCA
python -m pip install -r requirements.txt
python main.py
```
Refactoriza `tasks/pca_challenge.py` en funciones, e invócalas desde `src/flow/pipeline_pca.py`.
