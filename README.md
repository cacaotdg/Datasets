# Primero el codigo para clonar el repositorio

!pip install -q ultralytics albumentations
import os
!git clone https://github.com/cacaotdg/Datasets.git
%cd Datasets

#Ejecutar primero el entrenamiento.py para el entrenamiento del modelo con:

!python entrenamiento.py

#Luego ejecutar el metricas.py para ver los resultados con:
!python metricas.py
