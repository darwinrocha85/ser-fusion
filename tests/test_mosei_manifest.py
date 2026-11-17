"""Tests del manifest de CMU-MOSEI. Se saltan si los datos no están descargados."""
import numpy as np
import pandas as pd
import pytest

from serfusion.data.mosei_manifest import EMOTIONS, memberships
from serfusion.paths import manifest_dir

MANIFEST = manifest_dir("mosei") / "manifest.csv"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="manifest de MOSEI no generado")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(MANIFEST)


def test_conteos_del_archivo_oficial(df):
    # Cifras del archivo de etiquetas verificado por SHA-256 (2026-11-17).
    assert df.video_id.nunique() == 3293
    assert len(df) == 23259


def test_folds_oficiales(df):
    videos = df.groupby("split").video_id.nunique()
    assert videos["train"] == 2249
    assert videos["valid"] == 300
    assert videos["test"] == 676      # 678 en el SDK; 2 no tienen etiquetas
    assert videos["unassigned"] == 68


def test_ningun_video_en_dos_splits(df):
    assert (df.groupby("video_id").split.nunique() == 1).all()


def test_ids_unicos(df):
    assert df.segment_id.is_unique


def test_intensidades_validas(df):
    vals = df[EMOTIONS].to_numpy()
    assert vals.min() >= 0 and vals.max() <= 3
    # Las intensidades son medias de anotadores en pasos de 1/6.
    assert np.allclose(vals * 6, np.round(vals * 6), atol=1e-4)


def test_pertenencias_en_rango(df):
    mu = memberships(df)
    assert mu.shape == (len(df), 6)
    assert mu.min() >= 0 and mu.max() <= 1


def test_intervalos(df):
    assert (df.end > df.start).all()
