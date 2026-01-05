# Cost Analysis: Model vs LLM vs Hybrid

## Overview

This document compares the costs and trade-offs of three classification approaches:
1. **Model Only**: Fine-tuned Transformer (DistilBERT)
2. **LLM Only**: Claude/GPT for all classifications
3. **Hybrid**: Model with LLM fallback for low-confidence predictions

## Cost Comparison

### Per-Request Costs (Approximate)

| Approach | Cost per Request | Latency | Accuracy |
|----------|-----------------|---------|----------|
| Model (CPU) | $0.0001 | ~50-100ms | ~80-85% |
| Model (GPU) | $0.00005 | ~10-20ms | ~80-85% |
| Claude Haiku | $0.001-0.01 | ~500-1500ms | ~90-95% |
| GPT-4o-mini | $0.0005-0.005 | ~500-1500ms | ~90-95% |
| **Hybrid (75% model)** | **~$0.003** | ~150ms avg | ~88-92% |

### Monthly Cost Projection (10,000 requests/day)

| Approach | Monthly Cost | Notes |
|----------|-------------|-------|
| Model Only | $30 | Server costs only |
| LLM Only (Haiku) | $1,500-3,000 | API costs |
| LLM Only (GPT-4o-mini) | $750-1,500 | API costs |
| **Hybrid** | **$150-450** | 75% model, 25% LLM |

## Hybrid Approach Benefits

### Cost Savings
- If 75% of requests are handled by the model (confidence >= 0.75):
  - 75% × $0.0001 + 25% × $0.005 = $0.00133 per request
  - **vs $0.005 for LLM-only = 73% savings**

### When to Use Each

```
┌─────────────────────────────────────────────────────────┐
│                    Decision Matrix                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  High Volume + Low Budget → Model Only                  │
│  Low Volume + High Accuracy → LLM Only                  │
│  High Volume + High Accuracy → Hybrid                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Threshold Tuning

The confidence threshold determines when to use LLM fallback:

| Threshold | Model % | LLM % | Cost Reduction | Accuracy Impact |
|-----------|---------|-------|----------------|-----------------|
| 0.90 | ~50% | ~50% | ~50% | Highest accuracy |
| 0.80 | ~65% | ~35% | ~65% | High accuracy |
| **0.75** | **~75%** | **~25%** | **~73%** | **Balanced** |
| 0.70 | ~85% | ~15% | ~80% | Some accuracy loss |
| 0.60 | ~95% | ~5% | ~90% | More accuracy loss |

**Recommendation**: Start with 0.75 and adjust based on:
- Actual model performance on your data
- Business requirements for accuracy
- Budget constraints

## Monitoring Metrics

The API provides these metrics via `/stats`:

```json
{
  "total_requests": 10000,
  "model_decisions": 7500,
  "llm_fallbacks": 2500,
  "model_ratio": 0.75,
  "llm_ratio": 0.25,
  "total_cost_usd": 13.25,
  "avg_model_latency_ms": 85.5,
  "avg_llm_latency_ms": 950.2
}
```

## Recommendations for Production

1. **Monitor model_ratio**: If < 0.70, consider:
   - Retraining the model with more data
   - Lowering the threshold (accept more model predictions)

2. **Monitor LLM costs**: Set alerts if daily cost exceeds budget

3. **A/B testing**: Compare hybrid vs LLM-only accuracy on a sample

4. **Caching**: Consider caching LLM responses for repeated queries

## LLM Provider Comparison

| Provider | Model | Input Cost/1M | Output Cost/1M | Best For |
|----------|-------|---------------|----------------|----------|
| Anthropic | Claude 3 Haiku | $0.25 | $1.25 | Balanced |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 | Complex tasks |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 | Cost-effective |
| OpenAI | GPT-4o | $2.50 | $10.00 | Highest accuracy |

**Recommendation**: Start with GPT-4o-mini or Claude Haiku for classification fallback.
