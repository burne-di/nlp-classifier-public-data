# NLP Classifier - Hybrid Architecture

Este proyecto implementa un sistema de clasificación de texto híbrido que combina la velocidad de un modelo Transformer local con la precisión de un LLM (Large Language Model).

## 🧠 ¿Cómo funciona?

El sistema utiliza una arquitectura **Híbrida (Hybrid Classifier)** diseñada para optimizar costos y precisión:

1.  **Fast Path (Modelo Local)**: El texto ingresa primero a un modelo Transformer (basado en BERT/RoBERTa) entrenado localmente.
    *   ✅ **Ventaja**: Extremadamente rápido (~10ms) y muy barato.
    *   ❌ **Desventaja**: Puede fallar en casos complejos o ambiguos.
2.  **Confidence Check**: El sistema evalúa la "confianza" de la predicción del modelo local.
    *   Si `confianza > umbral` (ej. 0.75): Se devuelve la respuesta del modelo local.
    *   Si `confianza < umbral`: Se activa el "Slow Path".
3.  **Slow Path (Fallback a LLM)**: Se envía el texto a un LLM (ej. GPT-4, Claude) para que lo clasifique.
    *   ✅ **Ventaja**: Alta capacidad de razonamiento y comprensión de contexto.
    *   ❌ **Desventaja**: Más lento y costoso.

Esta arquitectura permite que el 90% de las peticiones sean resueltas por el modelo rápido (gratis), usando el LLM solo para los casos difíciles.

---

## 📂 Estructura del Proyecto

### `packages/classifier_core/`
Este es el "cerebro" del proyecto. Es una librería de Python instalable que contiene toda la lógica de negocio.

*   `hybrid_classifier.py`: **Core del sistema**. Contiene la clase `HybridClassifier` que implementa la lógica de decisión explicada arriba (Modelo vs LLM).
*   `model.py`: Define la clase `TextClassifier` que envuelve el modelo Transformer (Hugging Face) para realizar predicciones.
*   `llm_client.py`: Gestiona la conexión con proveedores de LLM (OpenAI, Anthropic).
*   `data.py`: Funciones para cargar y preprocesar los datos de texto.
*   `train.py`: Lógica de entrenamiento basada en PyTorch/Transformers.

### `services/api/`
Es la interfaz externa del sistema. Expone la funcionalidad a través de una API REST usando **FastAPI**.

*   `main.py`: Punto de entrada de la API. Define los endpoints:
    *   `POST /predict`: Usa solo el modelo local.
    *   `POST /predict/hybrid`: Usa la lógica híbrida.
    *   `GET /stats`: Muestra métricas de uso (cuántas veces se usó el LLM, costos ahorrados, etc.).
*   `schemas.py`: Define los modelos de datos (Pydantic) para las peticiones y respuestas (ej. qué formato tiene el JSON de entrada).
*   `deps.py`: Inyección de dependencias para cargar el modelo una sola vez al iniciar la API.

### `scripts/`
Scripts ejecutables para el ciclo de vida de Machine Learning (MLOps).

*   `prepare_data.py`: Limpia y prepara los datasets crudos.
*   `train.py`: Entrena un nuevo modelo Transformer y lo guarda en `models/`.
*   `evaluate.py`: Mide la precisión del modelo en el set de prueba.

### `models/` y `mlruns/`
*   `models/`: Almacena los artefactos de los modelos entrenados (pesos, tokenizadores).
*   `mlruns/`: Directorio de **MLflow** para traquear experimentos (métricas de entrenamiento, hiperparámetros).

### `docker-compose.yml`
Orquesta los servicios para levantar todo el entorno con un solo comando:
1.  **api**: Levanta el servidor FastAPI en el puerto 8000.
2.  **mlflow**: Levanta la interfaz de MLflow en el puerto 5000 para visualizar experimentos.

---

## 🚀 Guía de Inicio

### 1. Requisitos
*   Docker y Docker Compose
*   (Opcional) Python 3.11 para desarrollo local

### 2. Ejecutar con Docker
Para levantar la API y MLflow:

```bash
docker-compose up --build
```

La API estará disponible en `http://localhost:8000/docs` (Swagger UI).

### 3. Desarrollo Local
Si quieres entrenar o modificar el código:

```bash
# Instalar dependencias
pip install -e .[dev]

# Entrenar modelo
python scripts/train.py

# Ejecutar API localmente
uvicorn services.api.main:app --reload
```

## 📊 Métricas y Monitoreo
El sistema rastrea automáticamente:
*   **Model Ratio**: Porcentaje de peticiones resueltas por el modelo local.
*   **LLM Ratio**: Porcentaje de peticiones que requirieron LLM.
*   **Ahorro de Costos**: Comparación estimada vs usar LLM para todo.
*   **Latencia**: Tiempo de respuesta promedio.

Endpoints útiles:
*   `GET /health`: Ver estado del sistema.
*   `GET /stats`: Ver métricas de rendimiento.
