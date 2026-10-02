"""Pipeline de datos: descarga, preparación y carga por lotes de las imágenes."""
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Dataset de Hugging Face: fotos de hojas de frijol en 3 clases
# (sana, mancha angular, roya). Se descarga solo la primera vez.
DATASET = "AI-Lab-Makerere/beans"

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_MUESTRA = RAIZ / "data" / "sample"
TAM_IMAGEN = 224


def cargar_dataset():
    """Descarga (o lee de la caché) el dataset y devuelve (dataset, nombres_de_clases)."""
    ds = load_dataset(DATASET)
    etiquetas = ds["train"].features["labels"].names
    return ds, etiquetas


def crear_transformaciones(procesador):
    """Transformaciones de imagen. En entrenamiento se añade 'aumento de datos'
    (recortes y giros aleatorios) para que el modelo generalice mejor."""
    normalizar = transforms.Normalize(procesador.image_mean, procesador.image_std)
    entrenamiento = transforms.Compose([
        transforms.RandomResizedCrop(TAM_IMAGEN, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizar,
    ])
    evaluacion = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(TAM_IMAGEN),
        transforms.ToTensor(),
        normalizar,
    ])
    return entrenamiento, evaluacion


def _aplicar(transformacion):
    def fn(lote):
        return {
            "pixel_values": [transformacion(img.convert("RGB")) for img in lote["image"]],
            "labels": lote["labels"],
        }
    return fn


def _juntar(ejemplos):
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in ejemplos]),
        "labels": torch.tensor([e["labels"] for e in ejemplos]),
    }


def crear_dataloaders(ds, procesador, tam_lote=32, max_ejemplos=None):
    """Devuelve los DataLoaders de entrenamiento, validación y prueba."""
    t_ent, t_eval = crear_transformaciones(procesador)
    particiones = {}
    for nombre, t in [("train", t_ent), ("validation", t_eval), ("test", t_eval)]:
        parte = ds[nombre].shuffle(seed=42)
        if max_ejemplos:
            parte = parte.select(range(min(max_ejemplos, len(parte))))
        particiones[nombre] = parte.with_transform(_aplicar(t))

    # num_workers=0 evita problemas de multiproceso en Windows
    return (
        DataLoader(particiones["train"], batch_size=tam_lote, shuffle=True, collate_fn=_juntar),
        DataLoader(particiones["validation"], batch_size=tam_lote, collate_fn=_juntar),
        DataLoader(particiones["test"], batch_size=tam_lote, collate_fn=_juntar),
    )


def guardar_muestras(ds, etiquetas, por_clase=2):
    """Guarda unas pocas imágenes de prueba en data/sample/ para probar la app."""
    CARPETA_MUESTRA.mkdir(parents=True, exist_ok=True)
    conteo = {i: 0 for i in range(len(etiquetas))}
    for ejemplo in ds["test"]:
        clase = ejemplo["labels"]
        if conteo[clase] < por_clase:
            conteo[clase] += 1
            ruta = CARPETA_MUESTRA / f"{etiquetas[clase]}_{conteo[clase]}.jpg"
            ejemplo["image"].convert("RGB").save(ruta)
        if all(c >= por_clase for c in conteo.values()):
            break
