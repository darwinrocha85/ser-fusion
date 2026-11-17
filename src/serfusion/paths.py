"""Rutas del proyecto. Los datos viven fuera del repositorio, en SER_DATA (por defecto C:/ser-data)."""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SER_DATA", "C:/ser-data"))


def raw_dir(dataset: str) -> Path:
    """Datos originales descargados. Inmutables: nunca se escriben salvo en la descarga."""
    return DATA_ROOT / "raw" / dataset


def manifest_dir(dataset: str) -> Path:
    return DATA_ROOT / "manifests" / dataset


def load_config(dataset: str) -> dict:
    with open(REPO_ROOT / "configs" / f"{dataset}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
