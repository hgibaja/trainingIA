"""Interfaz web para probar el modelo entrenado: sube una foto y te dice la clase.

Uso:
    python src/app.py      # luego abre http://127.0.0.1:7860
"""
import sys

import gradio as gr

from model import CARPETA_MODELO, cargar_entrenado, predecir
from pipeline import CARPETA_MUESTRA

if not (CARPETA_MODELO / "config.json").exists():
    sys.exit("No hay modelo entrenado todavía. Ejecuta primero: python src/train.py")

modelo, procesador = cargar_entrenado()


def clasificar(imagen):
    return predecir(imagen, modelo, procesador)


ejemplos = sorted(str(p) for p in CARPETA_MUESTRA.glob("*.jpg"))

demo = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="pil", label="Foto de una hoja de frijol"),
    outputs=gr.Label(num_top_classes=3, label="Predicción"),
    title="Clasificador de enfermedades en hojas de frijol",
    description="Modelo ResNet-18 ajustado con el dataset 'beans' de Hugging Face.",
    examples=ejemplos or None,
)

if __name__ == "__main__":
    demo.launch()
