# mi-proyecto-ia

Clasificador de imágenes entrenado con **fine-tuning**: se toma un modelo que ya sabe "ver"
(preentrenado con millones de fotos) y se le enseña una tarea nueva con pocos datos.

**Tarea:** detectar enfermedades en hojas de frijol → `angular_leaf_spot`, `bean_rust`, `healthy`.

## ¿De dónde se descargan el modelo y los datos?

Ambos se descargan automáticamente de **Hugging Face** la primera vez que se ejecuta el código:

| Qué | Fuente | Tamaño |
|---|---|---|
| Modelo base | [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) | ~45 MB |
| Dataset | [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | ~170 MB |

Se guardan en la caché `C:\Users\<usuario>\.cache\huggingface\` y no se vuelven a descargar.
El modelo **ya entrenado** por ti se guarda en `models/modelo-final/`.

## Estructura

```
mi-proyecto-ia/
├── data/sample/            # imágenes de ejemplo (se generan al entrenar)
├── notebooks/
│   └── exploracion.ipynb   # ver y entender los datos
├── src/
│   ├── pipeline.py         # carga y preparación de datos
│   ├── model.py            # modelo: crear, cargar y predecir
│   ├── train.py            # entrenamiento + evaluación + informe
│   └── app.py              # interfaz web para probar el modelo
├── reports/informe.pdf     # resultados (se genera al entrenar)
└── models/                 # modelo entrenado (se genera al entrenar)
```

## Cómo usarlo

```powershell
# 1. Crear entorno e instalar (solo la primera vez)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Entrenar (en CPU tarda unos minutos por época)
python src/train.py
python src/train.py --rapido      # prueba de humo en segundos
python src/train.py --epocas 5    # entrenar más

# 3. Probar el modelo en el navegador
python src/app.py                 # abre http://127.0.0.1:7860
```

## ¿Sin GPU?

Funciona en CPU, solo que más lento. Por eso se usa ResNet-18 (modelo pequeño).
Para modelos grandes puedes usar Google Colab, que tiene GPU gratis.

## Ideas para seguir aprendiendo

- Cambiar `MODELO_BASE` en `src/model.py` por otro modelo (p. ej. `google/vit-base-patch16-224`).
- Cambiar `DATASET` en `src/pipeline.py` por otro dataset de imágenes de Hugging Face.
- Jugar con `--lr` (tasa de aprendizaje) y `--epocas` y comparar los informes.
