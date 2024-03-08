import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

class NERTrainer:
    def __init__(self, model, train_dataset, eval_dataset, epochs=5, lr=5e-5):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=8)
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.epochs = epochs
        total_steps = len(self.train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {total_loss/len(self.train_loader):.4f}")
