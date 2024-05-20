# Document AI - Key Information Extraction System

Production-ready pipeline for **Key Information Extraction** from business documents using **LayoutLM-V2** and multi-modal NER models.

## Overview

Built to automate structured data extraction from business cards, invoices, and receipts — reducing manual input costs by **80%** (~200M KRW/year in production).

## Features

- Multi-modal document understanding (text + layout + image)
- Named Entity Recognition (NER) with LayoutLM-V2
- Fine-tuning pipeline for custom document types
- Batch inference with FastAPI REST endpoint
- Support for business cards, invoices, receipts

## Architecture

```
Document Input (Image + Text)
        │
        ▼
  LayoutLM-V2 Processor
        │
        ▼
  Token Classification (NER)
        │
        ▼
  Structured JSON Output
```

## Quick Start

```bash
pip install -r requirements.txt
python src/api.py
```

```python
from src.model import DocumentNERModel

model = DocumentNERModel("model/layoutlmv2-finetuned", labels=["NAME", "ORG", "EMAIL", "PHONE", "O"])
results = model.predict(image, words, boxes)
```

## Results

| Metric | Score |
|--------|-------|
| F1 Score | 0.923 |
| Precision | 0.941 |
| Recall | 0.906 |
| Latency | <120ms |

## Tech Stack

- **Model:** LayoutLM-V2 (Microsoft)
- **Framework:** PyTorch + HuggingFace Transformers
- **API:** FastAPI
- **Infra:** Docker, Kubernetes

## Project Structure

```
my-project/
├── src/
│   ├── model.py       # LayoutLM-V2 NER model
│   ├── trainer.py     # Fine-tuning trainer
│   ├── dataset.py     # Document dataset loader
│   └── api.py         # FastAPI inference server
├── configs/
│   └── train_config.yaml
└── README.md
```
