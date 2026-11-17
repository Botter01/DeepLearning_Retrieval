import torch
import math
import copy
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.loss import DPRLoss


def make_train_loader(split, tokenizer, batch_size=32, max_length=128):
    pairs = []
    for question, docs in zip(split['question'], split['text']):
        for doc in docs:
            pairs.append((question, doc))

    def collate(batch):
        queries, documents = zip(*batch)

        q = tokenizer(list(queries), padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        d = tokenizer(list(documents), padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        return {'query_ids': q['input_ids'], 'query_mask': q['attention_mask'],
                'document_ids': d['input_ids'], 'document_mask': d['attention_mask']}

    return DataLoader(pairs, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)


class Trainer:
    def __init__(self):
        self.criterion = DPRLoss()

    @torch.no_grad()
    def evaluate(self, query_encoder, document_encoder, loader, device):
        query_encoder.eval()
        document_encoder.eval()
        total = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            q = query_encoder(batch['query_ids'], batch['query_mask'])
            d = document_encoder(batch['document_ids'], batch['document_mask'])
            total += self.criterion(q, d).mean().item()
        return total / len(loader)

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module, train_loader, val_loader, epochs=20, lr=4e-4, warmup_ratio=0.05):
        #raise NotImplementedError()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_encoder.to(device)
        document_encoder.to(device)

        optimizer = torch.optim.AdamW(list(query_encoder.parameters()) + list(document_encoder.parameters()), lr=lr)

        total_steps = epochs * len(train_loader)
        warmup_steps = int(warmup_ratio * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        best_val = float('inf')
        best_state = None
        bad_epochs = 0
        patience=5
        min_delta=0.0001

        for epoch in range(epochs):
            query_encoder.train()
            document_encoder.train()
            total_loss = 0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                query_emb = query_encoder(batch['query_ids'], batch['query_mask'])
                document_emb = document_encoder(batch['document_ids'], batch['document_mask'])

                loss = self.criterion(query_emb, document_emb).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            val_loss = self.evaluate(query_encoder, document_encoder, val_loader, device)
            print(f'Epoch {epoch + 1}/{epochs}: loss {total_loss / len(train_loader):.4f} | val {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}')

            if val_loss < best_val - min_delta:
                best_val = val_loss
                bad_epochs = 0
                best_state = (copy.deepcopy(query_encoder.state_dict()),
                                  copy.deepcopy(document_encoder.state_dict()))
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f'Early stopping, best val: {best_val:.4f}')
                    break

        if best_state is not None:
            query_encoder.load_state_dict(best_state[0])
            document_encoder.load_state_dict(best_state[1])

        return query_encoder, document_encoder