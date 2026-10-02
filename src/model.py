"""Modelo: se parte de una red preentrenada de Hugging Face y se adapta a nuestras clases."""
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Modelo base descargado de https://huggingface.co/microsoft/resnet-18
# Es pequeño (~45 MB) y se puede entrenar en CPU. Con GPU puedes probar
# modelos más grandes, p. ej. "google/vit-base-patch16-224".
MODELO_BASE = "microsoft/resnet-18"

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_MODELO = RAIZ / "models" / "modelo-final"


def crear_modelo(etiquetas, nombre=MODELO_BASE):
    """Descarga el modelo base y reemplaza su última capa por una con nuestras clases."""
    id2label = dict(enumerate(etiquetas))
    label2id = {e: i for i, e in id2label.items()}
    modelo = AutoModelForImageClassification.from_pretrained(
        nombre,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # la capa final original tenía 1000 clases
    )
    procesador = AutoImageProcessor.from_pretrained(nombre)
    return modelo, procesador


def cargar_entrenado(carpeta=CARPETA_MODELO):
    """Carga el modelo que guardó train.py."""
    modelo = AutoModelForImageClassification.from_pretrained(carpeta)
    procesador = AutoImageProcessor.from_pretrained(carpeta)
    modelo.eval()
    return modelo, procesador


@torch.no_grad()
def predecir(imagen, modelo, procesador):
    """Devuelve {clase: probabilidad} para una imagen PIL."""
    entradas = procesador(images=imagen.convert("RGB"), return_tensors="pt")
    probs = modelo(**entradas).logits.softmax(-1)[0]
    return {modelo.config.id2label[i]: float(p) for i, p in enumerate(probs)}
