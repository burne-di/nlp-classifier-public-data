# Arquitectura del Sistema NLP Classifier

**Versión:** 0.2.0
**Última actualización:** Enero 2026

---

## Resumen Ejecutivo

**NLP Classifier** es un sistema de clasificación de texto híbrido diseñado para procesar noticias económicas y financieras en español. Combina la velocidad y eficiencia de costos de un modelo Transformer local (Fast Path) con la capacidad de razonamiento de un Large Language Model (Slow Path) para casos de baja confianza.

**Stack Tecnológico:**
- **Backend:** FastAPI, Python 3.11+
- **Model Core:** PyTorch, Transformers (Hugging Face)
- **LLM Providers:** Anthropic (Claude), Groq (Llama 3), OpenAI (GPT-4)
- **MLOps:** MLflow, Docker
- **Infrastructure:** GPU/CPU support

**Características Principales:**
- Arquitectura Híbrida (Modelo Local + LLM Fallback)
- Detección automática de confianza
- Estimación de costos en tiempo real
- Soporte multiproveedor de LLMs
- API REST con modos simple y batch

---

## Estructura del Proyecto

```
nlp-classifier-public-data/
├── packages/
│   └── classifier_core/          # Núcleo de lógica de ML
│       ├── __init__.py
│       ├── config.py             # Configuración y settings
│       ├── model.py              # Wrapper de Hugging Face
│       ├── hybrid_classifier.py  # Lógica de decisión híbrida
│       ├── llm_client.py         # Clientes para APIs de LLM
│       ├── data.py               # Procesamiento de datos y tokenización
│       ├── train.py              # Loop de entrenamiento
│       └── metrics.py            # Cálculo de métricas
├── services/
│   └── api/                      # API REST
│       ├── __init__.py
│       ├── main.py               # Aplicación FastAPI
│       ├── schemas.py            # Modelos Pydantic (Request/Response)
│       └── deps.py               # Inyección de dependencias
├── scripts/                      # Scripts de utilidad CLI
│   ├── train.py                  # Script de entrenamiento
│   ├── evaluate.py               # Script de evaluación
│   └── prepare_data.py           # Script de limpieza de datos
├── models/                       # Artefactos de modelos guardados
├── mlruns/                       # Tracking de MLflow
├── notebooks/                    # Notebooks de exploración
├── docker-compose.yml
└── pyproject.toml
```

---

## Componentes del Sistema

### 1. Configuración (`packages/classifier_core/config.py`)

**Propósito:** Gestión centralizada de configuración usando `pydantic-settings`.

**Responsabilidades:**
- Cargar variables de entorno (`.env`).
- Definir rutas de directorios importantes (`data/`, `models/`).
- Mapear etiquetas a IDs (`LABEL2ID`, `ID2LABEL`).
- Configurar hiperparámetros por defecto.

---

### 2. Modelo Local (`packages/classifier_core/model.py`)

**Propósito:** Manejo del modelo Transformer local (BERT/RoBERTa).

**Clases:**

| Clase | Descripción |
|-------|-------------|
| `TextClassifier` | Wrapper de alto nivel para inferencia. Maneja tokenización, movimiento a GPU/CPU y post-procesamiento de logits. |

**Funciones Clave:**
- `load_pretrained_model()`: Carga un modelo base para fine-tuning.
- `load_trained_model()`: Carga un modelo y tokenizador ya entrenados desde disco.
- `predict(text)`: Retorna etiqueta y confianza para un solo texto.
- `predict_batch(texts)`: Inferencia optimizada para listas de textos.

---

### 3. Cliente LLM (`packages/classifier_core/llm_client.py`)

**Propósito:** Abstracción unificada para diferentes proveedores de LLM.

**Diseño:**
Usa el patrón **Strategy** (o Factory) para intercambiar proveedores sin cambiar el código cliente.

**Proveedores Soportados:**
1.  **ClaudeClient (Anthropic):** Optimizado para `claude-3-haiku`.
2.  **OpenAIClient:** Compatible con OpenAI GPT-4o.
3.  **GroqClient:** Implementación de alta velocidad usando Llama 3 via Groq API.

**Cálculo de Costos:**
Cada cliente calcula el costo estimado de la llamada en USD basándose en los tokens de entrada y salida definidos en constantes (ej. `INPUT_COST_PER_1M`).

---

### 4. Clasificador Híbrido (`packages/classifier_core/hybrid_classifier.py`)

**Propósito:** Orquestador principal de la lógica de decisión "Fast Path vs Slow Path".

**Algoritmo de Decisión:**

```python
def classify(self, text):
    # Paso 1: Predicción Modelo Local
    result = self.model.predict(text)
    
    # Paso 2: Evaluación de Confianza
    if result.confidence >= umbral:
        return result (Source: "model")
        
    # Paso 3: Fallback a LLM
    if llm_enabled:
        llm_result = self.llm.classify(text)
        return llm_result (Source: "llm")
        
    return result (Con advertencia de baja confianza)
```

**Métricas y Estadísticas:**
Mantiene un objeto `ClassificationStats` que rastrea:
- Ratio de uso del modelo vs LLM.
- Costo total acumulado.
- Latencia promedio por fuente.

---

### 5. API REST (`services/api/`)

**Propósito:** Interfaz de comunicación externa construida con FastAPI.

**Endpoints Principales:**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/predict` | Inferencia simple (solo modelo local). Rápido y determinista. |
| `POST` | `/predict/hybrid` | Inferencia híbrida. Puede invocar al LLM si es necesario. Retorna metadatos de costo y fuente. |
| `POST` | `/predict/hybrid/batch` | Versión batch del endpoint híbrido. |
| `GET` | `/stats` | Retorna estadísticas de uso actuales (hit rate, costos). |
| `GET` | `/health` | Chequeo de estado del sistema y carga de modelos. |

**Gestión de Ciclo de Vida (`lifespan`):**
Carga el modelo pesado en memoria **una sola vez** al iniciar la aplicación, compartiendo la instancia entre peticiones para eficiencia.

---

## Flujo de Datos

```
┌─────────────────────────────────────────────────────────────┐
│                    API Request (Hybrid)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   User Layout                                               │
│       │                                                     │
│       ▼                                                     │
│   ┌───────────────┐                                         │
│   │ FastAPI Router│                                         │
│   └──────┬────────┘                                         │
│          │                                                  │
│          ▼                                                  │
│   ┌────────────────────┐                                    │
│   │ HybridClassifier   │                                    │
│   └──────┬─────────────┘                                    │
│          │                                                  │
│          ▼                                                  │
│   ┌────────────────────┐    Confianza > 0.75?               │
│   │ Local Transformer  │──────────────────────────┐         │
│   │ (GPU/CPU)          │           SI             │         │
│   └─────────┬──────────┘                          │         │
│             │ NO                                  │         │
│             ▼                                     │         │
│   ┌────────────────────┐                          │         │
│   │ LLM Client         │                          │         │
│   │ (Anthropic/Groq)   │                          ▼         │
│   └─────────┬──────────┘                  ┌──────────────┐  │
│             │                             │  Response    │  │
│             └────────────────────────────▶│  JSON        │  │
│                                           └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuración de Producción

### Docker Compose

El sistema está contenerizado para facilitar el despliegue.

```yaml
services:
  api:
    image: nlp-classifier-api
    ports: ["8000:8000"]
    environment:
      - CONFIDENCE_THRESHOLD=0.75
      - LLM_PROVIDER=groq
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./models:/app/models:ro  # Comparte modelos entrenados

  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports: ["5000:5000"]
    command: mlflow server ...
```

### Variables de Entorno Clave

- `CONFIDENCE_THRESHOLD`: (float) Umbral 0.0-1.0 para activar el LLM.
- `LLM_PROVIDER`: `anthropic`, `openai`, `groq`.
- `MODEL_PATH`: Ruta al directorio del modelo entrenado.

---

## Métricas de Performance Estimadas

| Métrica | Local (Transformer) | LLM (Groq Llama 3) | LLM (Claude Haiku) |
|---------|---------------------|--------------------|--------------------|
| Latencia| ~20-50ms            | ~500ms - 1s        | ~1s - 2s           |
| Costo   | Despreciable (CPU)  | Muy bajo           | Medio              |
| Precisión| Alta (en dominio)  | Muy Alta (General) | Muy Alta (General) |

---

## Patrones de Diseño Utilizados

1.  **Strategy Pattern**: En `llm_client.py` para intercambiar proveedores de IA.
2.  **Singleton (via Dependency Injection)**: En FastAPI para mantener el modelo cargado en memoria.
3.  **Fallback Pattern**: En `HybridClassifier` para degradar elegantemente del modelo local al LLM.
4.  **Repository/DAO**: Implícito en `data.py` (Dataset loading).

