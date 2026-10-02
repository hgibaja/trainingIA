"""Entrenamiento: ajusta (fine-tuning) el modelo preentrenado a nuestras imágenes.

Uso:
    python src/train.py                 # entrenamiento completo (3 épocas)
    python src/train.py --epocas 5      # más épocas
    python src/train.py --rapido        # prueba rápida con pocos datos
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")  # genera gráficos sin abrir ventanas
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from model import CARPETA_MODELO, MODELO_BASE, crear_modelo
from pipeline import DATASET, RAIZ, cargar_dataset, crear_dataloaders, guardar_muestras

RUTA_INFORME = RAIZ / "reports" / "informe.pdf"


def evaluar(modelo, loader, dispositivo):
    """Calcula pérdida y precisión sobre datos que el modelo no usa para aprender."""
    modelo.eval()
    perdida_total, predicciones, reales = 0.0, [], []
    with torch.no_grad():
        for lote in loader:
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            salida = modelo(**lote)
            perdida_total += salida.loss.item() * len(lote["labels"])
            predicciones += salida.logits.argmax(-1).tolist()
            reales += lote["labels"].tolist()
    precision = sum(p == r for p, r in zip(predicciones, reales)) / len(reales)
    return perdida_total / len(reales), precision, predicciones, reales


def entrenar(args):
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}")

    print(f"Cargando dataset {DATASET}...")
    ds, etiquetas = cargar_dataset()
    guardar_muestras(ds, etiquetas)
    print(f"Clases: {etiquetas}")

    print(f"Cargando modelo base {MODELO_BASE}...")
    modelo, procesador = crear_modelo(etiquetas)
    modelo.to(dispositivo)

    max_ejemplos = 64 if args.rapido else None
    ent, val, prueba = crear_dataloaders(ds, procesador, args.tam_lote, max_ejemplos)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr)

    historial = []
    inicio = time.time()
    for epoca in range(1, args.epocas + 1):
        # --- Fase de aprendizaje ---
        modelo.train()
        perdida_ent = 0.0
        for i, lote in enumerate(ent, 1):
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            salida = modelo(**lote)       # 1. el modelo predice y calcula su error (loss)
            salida.loss.backward()        # 2. calcula cómo cambiar cada peso para reducir el error
            optimizador.step()            # 3. ajusta los pesos
            optimizador.zero_grad()
            perdida_ent += salida.loss.item()
            print(f"\r  Época {epoca}/{args.epocas} - lote {i}/{len(ent)} - loss {salida.loss.item():.3f}", end="")

        # --- Fase de validación ---
        perdida_val, precision_val, _, _ = evaluar(modelo, val, dispositivo)
        historial.append({
            "epoca": epoca,
            "perdida_entrenamiento": perdida_ent / len(ent),
            "perdida_validacion": perdida_val,
            "precision_validacion": precision_val,
        })
        print(f"\n  -> validación: loss {perdida_val:.3f} | precisión {precision_val:.1%}")

    minutos = (time.time() - inicio) / 60
    print(f"Entrenamiento terminado en {minutos:.1f} min")

    # --- Evaluación final con datos de prueba (nunca vistos) ---
    _, precision_prueba, preds, reales = evaluar(modelo, prueba, dispositivo)
    print(f"Precisión en prueba: {precision_prueba:.1%}")

    CARPETA_MODELO.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(CARPETA_MODELO)
    procesador.save_pretrained(CARPETA_MODELO)
    metricas = {
        "modelo_base": MODELO_BASE,
        "dataset": DATASET,
        "epocas": args.epocas,
        "minutos": round(minutos, 1),
        "dispositivo": dispositivo,
        "precision_prueba": precision_prueba,
        "historial": historial,
    }
    (CARPETA_MODELO / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    print(f"Modelo guardado en {CARPETA_MODELO}")

    generar_informe(metricas, etiquetas, reales, preds)
    print(f"Informe generado en {RUTA_INFORME}")


def generar_informe(metricas, etiquetas, reales, preds):
    """Crea reports/informe.pdf con curvas de entrenamiento y matriz de confusión."""
    RUTA_INFORME.parent.mkdir(parents=True, exist_ok=True)
    h = metricas["historial"]
    epocas = [x["epoca"] for x in h]
    etiquetas_usadas = list(range(len(etiquetas)))

    with PdfPages(RUTA_INFORME) as pdf:
        # Página 1: resumen y curvas
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Informe de entrenamiento", fontsize=18, weight="bold")
        resumen = (
            f"Modelo base: {metricas['modelo_base']}\n"
            f"Dataset: {metricas['dataset']}\n"
            f"Clases: {', '.join(etiquetas)}\n"
            f"Épocas: {metricas['epocas']}   |   Dispositivo: {metricas['dispositivo']}"
            f"   |   Tiempo: {metricas['minutos']} min\n\n"
            f"Precisión en datos de prueba: {metricas['precision_prueba']:.1%}"
        )
        fig.text(0.08, 0.83, resumen, fontsize=11, va="top", family="monospace")

        ax1 = fig.add_axes([0.1, 0.45, 0.8, 0.25])
        ax1.plot(epocas, [x["perdida_entrenamiento"] for x in h], "o-", label="entrenamiento")
        ax1.plot(epocas, [x["perdida_validacion"] for x in h], "o-", label="validación")
        ax1.set_title("Pérdida (loss) por época — más bajo es mejor")
        ax1.set_xlabel("época")
        ax1.set_xticks(epocas)
        ax1.legend()

        ax2 = fig.add_axes([0.1, 0.08, 0.8, 0.25])
        ax2.plot(epocas, [x["precision_validacion"] for x in h], "o-", color="green")
        ax2.set_title("Precisión en validación por época — más alto es mejor")
        ax2.set_xlabel("época")
        ax2.set_xticks(epocas)
        ax2.set_ylim(0, 1)
        pdf.savefig(fig)
        plt.close(fig)

        # Página 2: matriz de confusión y reporte por clase
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Resultados en datos de prueba", fontsize=16, weight="bold")
        ax = fig.add_axes([0.15, 0.45, 0.7, 0.42])
        cm = confusion_matrix(reales, preds, labels=etiquetas_usadas)
        ConfusionMatrixDisplay(cm, display_labels=etiquetas).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Matriz de confusión (filas = real, columnas = predicho)")
        reporte = classification_report(
            reales, preds, labels=etiquetas_usadas, target_names=etiquetas, zero_division=0
        )
        fig.text(0.1, 0.35, reporte, fontsize=10, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el clasificador de imágenes")
    parser.add_argument("--epocas", type=int, default=3)
    parser.add_argument("--tam-lote", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4, help="tasa de aprendizaje")
    parser.add_argument("--rapido", action="store_true", help="prueba rápida con pocos datos")
    entrenar(parser.parse_args())
