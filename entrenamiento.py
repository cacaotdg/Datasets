# metricas.py

import os
from ultralytics import YOLO

dataset_path = "/content/Datasets/escenas"
data_path = os.path.join(dataset_path, "PCA_open", "data.yaml")

# Ruta donde se guardó el modelo entrenado
model_path = "/content/Datasets/runs/detect/train/weights/best.pt"

print(f"Dataset: {data_path}")
print(f"Cargando modelo entrenado desde: {model_path}")

# --- EVALUACIÓN ---
model = YOLO(model_path)
print("Modelo cargado correctamente, comenzando evaluación...")

metrics = model.val(data=data_path)

# --- CÁLCULO DE MÉTRICAS ---
iou_50 = metrics.box.map50  # mAP@0.5
precision = metrics.box.mp  # Precisión
recall = metrics.box.mr     # Recall
f1_score = 2 * (precision * recall) / (precision + recall)

# --- RESULTADOS ---
print("\nResultados del modelo:")
print(f"IoU@0.5 (mAP@0.5): {iou_50:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
