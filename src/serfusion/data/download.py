"""Descarga verificada de los archivos fuente de un dataset.

Uso:
    python -m serfusion.data.download mosei                  # todos los archivos del config
    python -m serfusion.data.download mosei labels words     # solo algunos

Garantías:
- Se fija la revisión (commit) del espejo y se descarga siempre de ese commit.
- Cada archivo se verifica contra el SHA-256 completo publicado por el espejo y contra
  el prefijo anotado en el config. Si no coincide, el archivo se borra y se aborta.
- Al terminar se escribe raw/<dataset>/SOURCE.json con commit, hashes, tamaños y fecha,
  y los archivos quedan en solo lectura.
"""
import hashlib
import json
import os
import stat
import sys
import urllib.request
from datetime import datetime, timezone

import requests
from tqdm import tqdm

from serfusion.paths import load_config, raw_dir

HF_API = "https://huggingface.co/api/datasets"
HF_WEB = "https://huggingface.co/datasets"
CHUNK = 8 * 1024 * 1024


def _repo_id(mirror_url: str) -> str:
    return mirror_url.rstrip("/").split("/datasets/")[1]


def resolve_revision(repo: str, revision: str) -> str:
    r = requests.get(f"{HF_API}/{repo}/revision/{revision}", timeout=60)
    r.raise_for_status()
    return r.json()["sha"]


def remote_files(repo: str, commit: str) -> dict:
    """path -> {'size', 'sha256'} según el árbol del commit fijado."""
    r = requests.get(f"{HF_API}/{repo}/tree/{commit}", params={"recursive": "true"}, timeout=60)
    r.raise_for_status()
    out = {}
    for item in r.json():
        if item["type"] == "file" and item.get("lfs"):
            out[item["path"]] = {"size": item["lfs"]["size"], "sha256": item["lfs"]["oid"]}
    return out


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest, expected_size: int) -> None:
    """Descarga con reanudación (Range) a un .part y lo renombra al terminar."""
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    if done > expected_size:
        part.unlink()
        done = 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        if done and r.status_code != 206:  # el servidor ignoró el Range: empezar de cero
            done = 0
        mode = "ab" if done else "wb"
        with open(part, mode) as f, tqdm(total=expected_size, initial=done, unit="B",
                                         unit_scale=True, desc=dest.name) as bar:
            for block in r.iter_content(CHUNK):
                f.write(block)
                bar.update(len(block))
    part.rename(dest)


def main(dataset: str, keys: list[str]) -> None:
    cfg = load_config(dataset)
    src = cfg["source"]
    repo = _repo_id(src["mirror"])
    out_dir = raw_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_json = out_dir / "SOURCE.json"
    record = json.loads(source_json.read_text(encoding="utf-8")) if source_json.exists() else {}

    # Revisión fija: si ya hay una registrada se reutiliza, para no mezclar commits.
    commit = record.get("commit") or resolve_revision(repo, src["revision"])
    remote = remote_files(repo, commit)
    record.update({"mirror": src["mirror"], "commit": commit})
    record.setdefault("files", {})

    for key in keys or list(src["files"]):
        spec = src["files"][key]
        meta = remote[spec["path"]]
        if not meta["sha256"].startswith(spec["sha256"]):
            sys.exit(f"[{key}] el hash del espejo ({meta['sha256'][:16]}) no coincide con el config ({spec['sha256']})")
        dest = out_dir / os.path.basename(spec["path"])
        if not dest.exists():
            download(f"{HF_WEB}/{repo}/resolve/{commit}/{spec['path']}", dest, meta["size"])
        got = sha256_of(dest)
        if got != meta["sha256"]:
            os.chmod(dest, stat.S_IWRITE)
            dest.unlink()
            sys.exit(f"[{key}] SHA-256 incorrecto: {got} != {meta['sha256']}. Archivo borrado.")
        os.chmod(dest, stat.S_IREAD)  # solo lectura: raw/ es inmutable
        record["files"][key] = {"file": dest.name, "size": meta["size"], "sha256": got,
                                "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"[{key}] OK  {dest.name}  sha256={got[:16]}…")
        source_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # Folds oficiales del SDK de CMU (fuente oficial, independiente del espejo).
    folds_cfg = cfg.get("official_folds")
    if folds_cfg:
        folds_path = out_dir / "cmu_mosei_std_folds.py"
        if not folds_path.exists():
            urllib.request.urlretrieve(folds_cfg["url"], folds_path)
            os.chmod(folds_path, stat.S_IREAD)
        record["official_folds"] = {"url": folds_cfg["url"], "sha256": sha256_of(folds_path)}
        source_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[folds] OK  {folds_path.name}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
