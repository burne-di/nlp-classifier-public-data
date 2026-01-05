# Dataset Sources

## Primary Dataset: df_total.csv

### Description
Dataset de noticias económicas y financieras en español, clasificadas en 7 categorías.

### Source
- **La República** (Colombia): https://www.larepublica.co
- **BBVA Research**: https://www.bbva.com/es

### Statistics
- **Total records**: 1,217
- **Language**: Spanish
- **Format**: CSV

### Labels Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Macroeconomia | 340 | 27.94% |
| Alianzas | 247 | 20.30% |
| Innovacion | 195 | 16.02% |
| Regulaciones | 142 | 11.67% |
| Sostenibilidad | 137 | 11.26% |
| Otra | 130 | 10.68% |
| Reputacion | 26 | 2.14% |

### Schema

| Field | Type | Description |
|-------|------|-------------|
| url | string | Source URL of the article |
| news | string | Full text content of the news article |
| Type | string | Category label |

### Processing Notes
- Texts shorter than 50 characters are filtered out
- Data is split 80/10/10 (train/val/test) with stratification
- No null values in the dataset

### License
Public data scraped from news websites for educational purposes.
