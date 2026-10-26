# Sistema de morfología matemática

Aplicación de escritorio en Python para aplicar operaciones de morfología matemática a imágenes, con vista previa de la imagen original y procesada, y métricas de comparación.

Trabajo final de la materia de Procesamiento de imágenes, ESCOM-IPN, desarrollado con apoyo de herramientas de IA.

## Operaciones disponibles

- Erosión y dilatación.
- Apertura y cierre, en dos versiones: combinando erosión y dilatación, y con `cv2.morphologyEx`.
- Gradiente morfológico, top-hat y black-hat.
- Elementos estructurantes rectangular, en cruz y elíptico, con tamaño y número de iteraciones configurables.

## Métricas de comparación

Al procesar una imagen, el sistema calcula el error cuadrático medio (MSE), la relación señal a ruido pico (PSNR) y el índice de similitud estructural (SSIM) entre la imagen original y la procesada.

## Requisitos

- Python 3.9 o superior, con soporte para Tkinter.
- Dependencias listadas en `requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
python sistema_morfologia.py
```

## Autor

Sergio Alejandro Vera Martínez, ESCOM-IPN.
