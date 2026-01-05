# Arquitectura del Sistema NLP Classifier

**Version:** 0.2.0  
**Ultima actualizacion:** Enero 2026

---

## Resumen Ejecutivo

**NLP Classifier** es un sistema de clasificacion hibrida para noticias economicas y financieras en espanol. Combina un modelo Transformer local (baja latencia y bajo costo) con un LLM externo como fallback cuando la confianza del modelo es baja. El objetivo es maximizar precision sin disparar costos operativos.

**Stack Tecnologico:**
- **Backend:** FastAPI, Pydantic, Python 3.11+
- **Modelo local:** PyTorch + Transformers (Hugging Face)
- **LLM Providers:** Groq, OpenAI, Anthropic (opcionales)
- **MLOps:** MLflow + artefactos locales
- **Contenedores:** Docker + Docker Compose

**Caracteristicas Principales:**
- Arquitectura hibrida (modelo local + LLM fallback)
- Umbral de confianza configurable
- Estimacion de costo por request en modo hibrido
- Endpoints batch y single
- Training bloqueante via endpoint controlado

---

## Estructura del Proyecto

```
nlp-classifier-public-data/
  .github/
    workflows/
      ci.yml
  data/
    raw/                      # CSV crudo (fuente)
    processed/
      train.csv
      val.csv
      test.csv
      extra_train.csv         # dataset extra opcional
  docs/
    cost_analysis.md
    dataset_sources.md
    decisions.md
    model_card.md
  mlruns/                     # MLflow artifacts (generado)
  models/                     # modelos entrenados (generado)
  notebooks/
  packages/
    classifier_core/
      __init__.py
      config.py               # Settings, labels, paths
      data.py                 # carga y tokenizacion
      model.py                # wrapper HF
      metrics.py              # metricas
      train.py                # training utilities
      llm_client.py           # LLM providers
      hybrid_classifier.py    # decision hibrida
  scripts/
    prepare_data.py           # limpieza y split
    train.py                  # entrenamiento con MLflow
    evaluate.py               # evaluacion offline
  services/
    api/
      deps.py                 # DI, carga de modelo
      main.py                 # FastAPI app
      schemas.py              # Pydantic schemas
  tests/
    test_api.py
    test_data.py
  .env
  .env.example
  Dockerfile
  docker-compose.yml
  pyproject.toml
  README.md
  ARCHITECTURE.md
```

---

## Componentes del Sistema

### 1. Configuracion (packages/classifier_core/config.py)
**Proposito:** gestion centralizada de settings y mapeo de labels.  
**Librerias:** pydantic_settings, functools.lru_cache  
**Clase principal:** `Settings`

```python
class Settings(BaseSettings):
    model_name: str = "distilbert-base-multilingual-cased"
    max_length: int = 512
    num_labels: int = 7
    batch_size: int = 16
    learning_rate: float = 2e-5
```

**Integracion:** toda la app accede via `get_settings()` (singleton).

---

### 2. Datos y Split (data.py + scripts/prepare_data.py)
**Proposito:** limpiar, validar y generar splits estratificados.  
**Entradas:** CSV crudo con columnas `news` y `Type`.  
**Salida:** `train.csv`, `val.csv`, `test.csv`.  
**Nota:** si existe `extra_train.csv`, se concatena al train en runtime.

---

### 3. Modelo Local (model.py)
**Proposito:** inferencia con Transformer fine-tuned.  
**Clase principal:** `TextClassifier`  
**Responsabilidades:**
- Tokenizar a `max_length` fijo
- Ejecutar inferencia en CPU/GPU
- Calcular softmax y confianza

---

### 4. Entrenamiento (train.py + scripts/train.py)
**Proposito:** fine-tuning del modelo con HF Trainer y early stopping.  
**Algoritmo:**
1. Carga dataset tokenizado
2. Entrena con `TrainingArguments`
3. Evalua en validation y test
4. Guarda `final_model/` y registra en MLflow

---

### 5. Metricas (metrics.py)
**Proposito:** calcular accuracy y macro F1/precision/recall.  
**Uso:** HF Trainer y scripts de evaluacion offline.

---

### 6. LLM Client (llm_client.py)
**Proposito:** interfaz unificada para proveedores LLM.  
**Clases:** `ClaudeClient`, `OpenAIClient`, `GroqClient`.  
**Detalles:**
- Prompt fijo con 7 categorias
- Parseo defensivo de etiquetas
- Estimacion de costo por tokens

---

### 7. Clasificador Hibrido (hybrid_classifier.py)
**Proposito:** decidir si usar modelo local o LLM.  
**Algoritmo (simplificado):**

```text
predict_local(text) -> (label, confidence)
if confidence >= threshold:
  return model result
if llm_enabled:
  return llm result
return model result (low confidence)
```

**Stats:** ratio modelo vs LLM, costo acumulado, latencias promedio.

---

### 8. API REST (services/api)
**Proposito:** interfaz externa de inferencia y entrenamiento.  
**Endpoints principales:**
- `POST /predict` (modelo local)
- `POST /predict/batch`
- `POST /predict/hybrid`
- `POST /predict/hybrid/batch`
- `GET /health`
- `GET /stats` + `POST /stats/reset`
- `POST /train` (bloqueante, con token)

**Lifespans:** carga el modelo una sola vez al inicio.

---

### 9. Tracking (mlruns/)
**Proposito:** MLflow registra parametros, metricas y artefactos.  
**Almacen:** local, sin dependencia cloud.

---

### 10. Docker y Compose
**Proposito:** estandarizar runtime y despliegue.  
**Servicios:** `api` + `mlflow`.

---

## Flujo de Datos End-to-End

### Entrenamiento (batch)
```
Raw CSV -> scripts/prepare_data.py -> data/processed/*.csv
                           |
                           v
                    scripts/train.py
                           |
        +------------------+------------------+
        |                                     |
     mlruns/ (MLflow)                 models/run_*/final_model
```

### Inferencia simple (sync)
```
Client -> POST /predict -> FastAPI -> TextClassifier -> label + confidence
```

### Inferencia hibrida (sync)
```
Client -> POST /predict/hybrid -> HybridClassifier
  -> local model (confidence)
     -> if >= threshold: response(model)
     -> else: response(LLM) if enabled
```

---

## Configuracion de Produccion

### Variables de Entorno (.env)
- `MODEL_PATH`: ruta del modelo cargado por la API
- `CONFIDENCE_THRESHOLD`: umbral de fallback
- `LLM_PROVIDER`: groq | openai | anthropic
- `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `TRAINING_ENABLED`: habilita `/train`
- `TRAINING_TOKEN`: token simple para `/train`
- Hiperparametros: `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, etc.

### Docker Compose (resumen)
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./models:/app/models:ro
  mlflow:
    image: mlflow/mlflow:latest
    ports: ["5000:5000"]
```

---

## Dependencias

### Core
- fastapi, uvicorn
- torch, transformers, datasets
- pydantic, pydantic-settings
- scikit-learn, pandas
- mlflow

### LLM Providers
- openai (compatible)
- anthropic

### Dev
- pytest, httpx
- ruff

---

## Seguridad y Validaciones
- Pydantic valida formatos y longitudes de input.
- `/train` requiere `TRAINING_ENABLED=true` y token opcional.
- No hay auth para inferencia ni rate limiting.
- Riesgos: prompt injection y uso no controlado de LLM.

---

## Observabilidad
- Logs en consola del contenedor.
- `/health` para estado del modelo.
- `/stats` para ratio de fallback y costos.
- MLflow para metricas y artefactos de training.

---

## Performance y Escalabilidad
- Cuello de botella: inferencia CPU y max_length=512.
- LLM fallback agrega latencia de red.
- Escalar x10: replicas horizontales y cache; separar training de API.

---

## Testing y Calidad
- `tests/test_api.py`: validaciones basicas de endpoints.
- `tests/test_data.py`: estructura de datos y labels.
- Faltantes: training end-to-end, LLM fallback, docker-compose.

---

## Archivos o Componentes Auxiliares
- `mlruns/` y `models/`: artefactos generados.
- `notebooks/`: exploracion, no runtime.
- `extra_train.csv`: dataset opcional para refuerzo.

---

## Conclusion
El sistema es un MVP solido con elementos de produccion (API, Docker, CI, MLflow). Destaca por el enfoque hibrido y control de costos, pero requiere endurecimiento para entornos abiertos (auth, rate limiting, aislamiento de training) y mas data para mejorar generalizacion.
