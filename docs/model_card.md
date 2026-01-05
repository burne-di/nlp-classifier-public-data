# Model Card: NLP Classifier

## Model Details

### Model Description
Fine-tuned transformer model for classifying Spanish financial/economic news into 7 categories.

### Base Model
- **Name**: distilbert-base-multilingual-cased
- **Architecture**: DistilBERT
- **Parameters**: ~135M
- **Languages**: Multilingual (104 languages including Spanish)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Max sequence length | 512 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |

## Intended Use

### Primary Use Case
Classification of Spanish news articles into economic/financial categories for content organization and analysis.

### Out-of-Scope Use
- Classification of non-Spanish text
- Real-time financial trading decisions
- Legal or medical advice classification

## Training Data

### Dataset
- **Source**: News articles from La República and BBVA Research
- **Size**: 1,217 samples
- **Split**: 80% train / 10% validation / 10% test

### Labels
1. Macroeconomia
2. Alianzas
3. Innovacion
4. Regulaciones
5. Sostenibilidad
6. Otra
7. Reputacion

## Evaluation

### Metrics
- **Primary metric**: F1 Macro (target >= 0.70)
- **Secondary metrics**: Accuracy, Precision, Recall

### Known Limitations
- Class imbalance: "Reputacion" class has only 26 samples (2.14%)
- Domain-specific: Trained on financial/economic news only
- Text length: Performance may vary on very short or very long texts

## Ethical Considerations

### Bias
- Training data is from specific news sources, may not represent all perspectives
- Spanish variety is primarily Latin American

### Privacy
- No personal data is used in training
- All training data is from public sources

## Usage

### API Endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "El banco central anunció nuevas regulaciones."}'
```

### Response
```json
{
  "label": "Regulaciones",
  "confidence": 0.92
}
```
