# Technical Decisions

## Architecture Decisions

### ADR-001: Base Model Selection
**Decision**: Use `distilbert-base-multilingual-cased` as the base model.

**Context**: Need a model that supports Spanish text classification with good performance/size trade-off.

**Alternatives Considered**:
- `bert-base-multilingual-cased`: Larger, slower inference
- `xlm-roberta-base`: Better multilingual performance but larger
- `dccuchile/bert-base-spanish-wwm-cased`: Spanish-only, less flexible

**Rationale**: DistilBERT offers 60% of BERT's size with 97% of performance. Good balance for production use.

### ADR-002: Framework Selection
**Decision**: Use Hugging Face Transformers with PyTorch backend.

**Rationale**:
- Industry standard for NLP
- Excellent documentation and community support
- Easy integration with MLflow
- Good production deployment options

### ADR-003: API Framework
**Decision**: Use FastAPI for the inference API.

**Rationale**:
- Modern async support
- Automatic OpenAPI documentation
- Pydantic integration for validation
- High performance

### ADR-004: Experiment Tracking
**Decision**: Use MLflow for experiment tracking.

**Rationale**:
- Open source
- Local-first (no cloud dependency)
- Model registry capabilities
- Good integration with Python ML ecosystem

## Data Decisions

### DDR-001: Minimum Text Length
**Decision**: Filter texts shorter than 50 characters.

**Rationale**: Very short texts don't provide enough context for meaningful classification.

### DDR-002: Class Imbalance Handling
**Decision**: Use stratified splits; consider class weights in future iterations.

**Rationale**:
- MVP approach: maintain natural distribution
- Future: implement oversampling or class weights if performance is poor on minority classes

### DDR-003: Data Split Ratio
**Decision**: 80/10/10 train/validation/test split.

**Rationale**: Standard split for datasets of this size (~1200 samples).

## Training Decisions

### TDR-001: Hyperparameters
**Decision**: Use standard fine-tuning hyperparameters.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 2e-5 | Standard for BERT fine-tuning |
| Batch size | 16 | Memory-efficient |
| Epochs | 3 | Sufficient for fine-tuning |
| Max length | 512 | Full model capacity |

### TDR-002: Early Stopping
**Decision**: Implement early stopping with patience=2.

**Rationale**: Prevent overfitting on small dataset.

### TDR-003: Best Model Selection
**Decision**: Select best model based on F1 Macro on validation set.

**Rationale**: F1 Macro is robust to class imbalance, preferred over accuracy.
