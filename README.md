# Análisis Predictivo de Precios de Vivienda en King County 🏙️

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-407487?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTAgMjUwIj48cGF0aCBmaWxsPSIjRkZGIiBkPSJNNjEuMyA7MS4xYy0xLjgtNC4zLTEuMy0xNi4zLjgxLTE5LjggMi4xLTMuNSAxMS43LTUuMiAxNy4zLTQuMiA1LjYgMSA5LjYgNC4xIDExLjggOC43IDIuMiA0LjYgMS43IDE2LjItLjgyIDIwLTIuNiAzLjctMTEuNyA1LjQtMTcuNCA0LjMtNS41LTEuMS05LjYtNC4yLTExLjctOC45em0xODcgMTQ3LjljLTEuMy0uNi0xLjMtMS42IDAtMi4yIDEwLjEtNC42IDE5LjEtMTIuNiAyNC4xLTIzLjMgNS0xMC44IDYtMjMuNyAzLjEtMzUuOS0yLjktMTIuMy05LjEtMjMuNS0xNy41LTMyLjgtOC40LTkuNC0xOC45LTE2LjctMzAuNS0yMS40LTExLjYtNC43LTI0LjEtNi40LTM2LjUtNC43LTEyLjQuNS0yNC41IDQuOS0zNSAxMi41LTEwLjUgNy42LTE5IDE4LTI0LjIgMjkuOGwtMTYuNyA1MS45Yy42IDEuMyAxLjYgMS4zIDIuMiAwIDEwLjgtMTUgMjQuMi0yNi41IDM5LjMtMzQuMSAxNS4xLTcuNiAzMS41LTExLjggNDguMi0xMi40IDE2LjctLjYgMzMuMyAyLjUgNDguNSA4LjcgMTUuMiA2LjIgMjguNyAxNS42IDM5LjggMjcuNiAxMS4xIDEyIDE5LjIgMjYuNSAyMy4zIDQxLjggNC4xIDE1LjMgNC4xIDMxLjYgMCA0Ny40bC0xOS4zLTIuMnoiLz48cGF0aCBmaWxsPSIjRkZGIiBkPSJNMTAwLjQgMTEuNmMtLjMtLjMtLjgtLjMtMS4xIDAgMS44IDIuNyAzLjMgNTguNCA0LjQgODEuMiAxLjEgMjIuOCAxLjcgNDUuNi0xLjEgNjMuNi0xLjMtMS0uNy0yLjMuOC0yLjlsMzMuMy0xMy4yYzE5LTcuNiAzOC0xNS4zIDU3LTIzTDIyNy4yIDc5YzItMS4zIDEuMy0zLjUtLjgtNC4ybC03Mi4yLTQ4LjljLTIuMS0xLjQtNC40LTEuNC02LjUtLjVsLTM3LjUgMTQuOGMtMi40IDEuMi0zLjIgMy42LTEuNiA1LThsMzAuOSA0My4xYzEuNyAyLjIgMS42IDUuNC0uMiA3LjRsLTQ5IDQ0LjJjLTEuOCAxLjYtNC42IDEuMy02LjEtLjdMMTAwLjQgMTEuNnoiLz48L3N2Zz4=)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat&logo=googlecolab&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

## Descripción General

Este proyecto presenta un pipeline completo de Machine Learning para predecir los precios de las viviendas en King County, USA. A partir de un conjunto de datos público, se realiza un preprocesamiento exhaustivo, se entrena un modelo de Regresión Lineal y se evalúa su rendimiento para establecer un baseline predictivo.

---

## 💾 Fuente de Datos

El conjunto de datos utilizado es **"House Sales in King County, USA"**. Contiene información sobre ventas de casas entre mayo de 2014 y mayo de 2015.

* **Origen:** Kaggle
* **Enlace de Descarga:** [https://www.kaggle.com/harlfoxem/housesalesprediction/download](https://www.kaggle.com/harlfoxem/housesalesprediction/download)

---

## ⚙️ Pipeline del Proyecto

El análisis sigue un flujo de trabajo estructurado para garantizar la reproducibilidad y la calidad del modelo:

1.  **Limpieza de Datos:**
    * Manejo de valores nulos.
    * Eliminación de características irrelevantes (`id`, `date`) que no aportan valor predictivo.
    * Corrección de datos erróneos (ej. viviendas con superficie igual a cero).

2.  **Ingeniería de Características (Feature Engineering):**
    * Aplicación de **One-Hot Encoding** en variables categóricas (`waterfront`, `view`, `condition`, `grade`) para su correcta interpretación por el modelo.

3.  **Preprocesamiento:**
    * Normalización de todas las características numéricas utilizando `MinMaxScaler` para estandarizar su escala y mejorar el rendimiento del modelo.

4.  **Modelado y Evaluación:**
    * Entrenamiento de un modelo de **Regresión Lineal** como baseline.
    * Evaluación del rendimiento utilizando las métricas **R-squared (R²)** y **Mean Squared Error (MSE)**.

---

## 🚀 Cómo Empezar

Para ejecutar este proyecto en tu entorno local, sigue estos pasos:

1.  **Clona el repositorio:**
    ```sh
    git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
    ```
2.  **Navega a la carpeta del proyecto:**
    ```sh
    cd TU_REPOSITORIO
    ```
3.  **Instala las dependencias:**
    *Se recomienda crear un entorno virtual.*
    ```sh
    pip install pandas scikit-learn seaborn numpy
    ```
4.  **Ejecuta el notebook:**
    Abre el archivo `.ipynb` en Google Colab, Jupyter Notebook o VS Code. Asegúrate de tener el archivo `kc_house_data.csv` en la misma carpeta para que el notebook pueda cargarlo.

---

## 📈 Resultados

El modelo baseline de Regresión Lineal alcanzó un **R² de 0.82**, indicando que es capaz de explicar el 82% de la varianza en los precios de las viviendas con las características proporcionadas.

Sin embargo, el Root Mean Squared Error (RMSE) de aproximadamente **$155,000** sugiere que, aunque es un buen punto de partida para entender las relaciones en los datos, se requieren modelos más avanzados para una precisión a nivel de producción.

---

## 📄 Licencia

Este proyecto está distribuido bajo la Licencia MIT.
