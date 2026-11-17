# Plan de trabajo — CMU-MOSEI multietiqueta difuso

Destino: artículo corto (4–5 págs.) en CoNCISa 2027, envío por CMT antes del **12 de enero de 2027**
(fecha límite oficial: 15 de enero).

## Pregunta
Cuando cada modalidad se resume en grados de pertenencia a 6 emociones, ¿basta un centroide difuso
sin parámetros entrenables para fusionarlas, o un modelo entrenable aporta una mejora significativa?

## Hecho (2026-11-17)
- [x] Descarga verificada por SHA-256 de los 5 `.csd` + folds oficiales del SDK de CMU.
- [x] `manifest.csv`: 23 259 segmentos; train 16 327 / valid 1871 / test 4662 / fuera de folds 399.
- [x] Tests del manifest (7).

## Pendiente, en orden
1. [ ] **Features por segmento.** Media de COVAREP (74D), FACET (35D) y GloVe (300D) dentro del intervalo
       de cada segmento. Reportar la cobertura por modalidad; sin rellenar ceros en silencio.
       Tests: alineación con el manifest y ausencia de NaN/inf.
2. [ ] **Unimodales.** Por modalidad, 6 pertenencias estimadas. Train: fuera de pliegue (5 folds agrupados
       por vídeo, semilla 42). Valid/test: modelo ajustado en todo train. Escalado solo en train.
       Test anti-fuga: ningún vídeo en train y predicción a la vez.
3. [ ] **Fusión (18D).** Media de pertenencias, centroide difuso c_e = Σ μ_e(x)·x / Σ μ_e(x) con coseno,
       LogReg / Ridge por emoción. Umbrales elegidos solo en valid.
4. [ ] **Evaluación en test, una sola vez.** Binaria (presente si > 0): exactitud ponderada y F1 por emoción.
       Difusa: MAE y correlación de pertenencias. IC por bootstrap (10 000) y McNemar por emoción.
       Dominancia de cada modalidad.
5. [ ] `verify.py` + `results/expected/` con tolerancia.
6. [ ] Redacción.

## Reglas
- Nada de lo archivado en `_legacy/` se usa.
- El test no se mira hasta el paso 4.
- No inventar cifras: todo número del artículo sale de un archivo de resultados con su commit.
