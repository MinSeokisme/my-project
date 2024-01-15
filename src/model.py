import torch
from transformers import LayoutLMv2ForTokenClassification, LayoutLMv2Processor
from typing import List, Dict, Any

class DocumentNERModel:
    """LayoutLM-V2 based NER model for key information extraction."""
    def __init__(self, model_path: str, labels: List[str]):
        self.processor = LayoutLMv2Processor.from_pretrained(model_path)
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(model_path, num_labels=len(labels))
        self.labels = labels
        self.id2label = {i: l for i, l in enumerate(labels)}
        self.model.eval()

    def predict(self, image, words, boxes):
        encoding = self.processor(image, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = self.model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        return [{"word": w, "entity": self.id2label.get(p, "O")} for w, p in zip(words, predictions[1:-1])]
