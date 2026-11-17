"""Manifest de CMU-MOSEI: una fila por segmento etiquetado, única fuente de etiquetas del proyecto.

Uso:
    python -m serfusion.data.mosei_manifest

Entradas (raw/mosei, verificadas por SHA-256 en la descarga):
- CMU_MOSEI_Labels.csd: por vídeo, intervalos (inicio, fin) y 7 columnas
  [sentiment, happy, sad, anger, surprise, disgust, fear].
- cmu_mosei_std_folds.py: folds oficiales del SDK de CMU (se leen con ast, sin ejecutarlos).

Salida: manifests/mosei/manifest.csv con
    segment_id, video_id, seg_index, start, end, split, sentiment,
    happy, sad, anger, surprise, disgust, fear     (intensidades originales, 0..3)

Las pertenencias difusas se derivan en el código como intensidad / 3; no se guardan aparte
para que haya una sola versión de la verdad.
"""
import ast
import json

import h5py
import numpy as np
import pandas as pd

from serfusion.paths import manifest_dir, raw_dir

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
LABEL_COLUMNS = ["sentiment"] + EMOTIONS
SPLITS = {"standard_train_fold": "train", "standard_valid_fold": "valid", "standard_test_fold": "test"}


def read_official_folds(path) -> dict[str, str]:
    """video_id -> split, leyendo las listas del archivo del SDK sin ejecutarlo."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    video_split = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            name = node.targets[0].id
            if name in SPLITS:
                for elt in node.value.elts:
                    vid = ast.literal_eval(elt)
                    if vid in video_split:
                        raise ValueError(f"vídeo {vid} aparece en dos folds oficiales")
                    video_split[vid] = SPLITS[name]
    return video_split


def build() -> pd.DataFrame:
    raw = raw_dir("mosei")
    video_split = read_official_folds(raw / "cmu_mosei_std_folds.py")
    with h5py.File(raw / "CMU_MOSEI_Labels.csd", "r") as f:
        root = f["All Labels"]
        names = json.loads(root["metadata"]["dimension names"][()][0])
        if names != LABEL_COLUMNS:
            raise ValueError(f"columnas de etiquetas inesperadas: {names}")
        rows = []
        for vid in root["data"]:
            feats = root["data"][vid]["features"][()]
            inter = root["data"][vid]["intervals"][()]
            for i, (lab, (start, end)) in enumerate(zip(feats, inter)):
                rows.append({"segment_id": f"{vid}[{i}]", "video_id": vid, "seg_index": i,
                             "start": float(start), "end": float(end),
                             "split": video_split.get(vid, "unassigned"),
                             **dict(zip(LABEL_COLUMNS, map(float, lab)))})
    return pd.DataFrame(rows).sort_values(["video_id", "seg_index"]).reset_index(drop=True)


def memberships(df: pd.DataFrame) -> np.ndarray:
    """Grados de pertenencia difusos (N, 6) en [0, 1]."""
    return df[EMOTIONS].to_numpy() / 3.0


def main() -> None:
    df = build()
    out = manifest_dir("mosei")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "manifest.csv", index=False)
    resumen = df.groupby("split").agg(videos=("video_id", "nunique"), segmentos=("segment_id", "size"))
    print(resumen.to_string())
    print(f"total: {df.video_id.nunique()} vídeos, {len(df)} segmentos -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
