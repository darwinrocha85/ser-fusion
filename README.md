# ser-fusion

Fusión tardía multimodal para reconocimiento de emociones con un pipeline **limpio, reproducible y verificable**.

Pregunta de partida: cuando cada modalidad se resume en un vector de probabilidades, ¿basta un clasificador
de centroide coseno sin parámetros para fusionarlas, o la ventaja de modelos más complejos depende del
dataset y de la representación?

## Principios

1. **Fuentes originales, nada heredado.** Los datos se descargan de su fuente, se verifican por SHA-256
   y la carpeta `raw/` es de solo lectura.
2. **Una sola fuente de etiquetas.** Cada dataset tiene un `manifest.csv` derivado de las anotaciones
   originales. Ningún archivo intermedio guarda etiquetas.
3. **Sin fuga por construcción.** Validación anidada; escalado y selección de modelo ajustados solo en train;
   probabilidades unimodales siempre fuera de pliegue.
4. **Verificable.** Configuración en YAML, semilla fija, entorno bloqueado (`uv.lock`), tests anti-fuga
   y comparación automática con resultados esperados.

## Uso

```bash
uv sync --python 3.12
uv run python -m serfusion.data.download mosei      # descarga verificada (~15 GB)
uv run pytest
```

Los datos van en `SER_DATA` (por defecto `C:/ser-data`), fuera del repositorio.

## Datos

| Dataset | Fuente | Estado |
|---|---|---|
| CMU-MOSEI | Espejo de los `.csd` oficiales en Hugging Face (el servidor de CMU no responde); folds oficiales del SDK de CMU | En curso |
| IEMOCAP | Pendiente de licencia oficial | Más adelante |
