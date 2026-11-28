import torch
import math
import copy
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from src.training.loss import DPRLoss
from src.retrievers.vector_database import VectorDatabase, encode
from src.evaluation.metrics import Evaluate


class QuestionDataset(Dataset):
    def __init__(self, questions, answers):
        self.items = [(q, a) for q, a in zip(questions, answers) if len(a) > 0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        question, answers = self.items[idx]
        return question, random.choice(answers)


class RetrievalEvaluator:
    def __init__(self, doc_texts, doc_ids, queries, tokenizer, k=10, max_length=512, bs=256):
        self.doc_texts = doc_texts
        self.doc_ids = doc_ids
        self.queries = queries
        self.tokenizer = tokenizer
        self.k = k
        self.max_length = max_length
        self.bs = bs

    def __call__(self, query_encoder, document_encoder, device):
        query_encoder.eval()
        document_encoder.eval()

        doc_emb = encode(self.doc_texts, document_encoder, self.tokenizer, device, bs=self.bs, max_length=self.max_length)
        q_emb = encode([q.getQuery() for q in self.queries], query_encoder, self.tokenizer, device, bs=self.bs, max_length=self.max_length)

        retrieved = VectorDatabase(doc_emb, self.doc_ids).search(q_emb, self.k)

        rels = [[doc_id in set(q.relevant_document_ids) for doc_id in row]
                for q, row in zip(self.queries, retrieved)]
        return Evaluate(rels, self.queries, k=self.k)


def make_train_loader(split, tokenizer, batch_size=32, max_length=512):
    pairs = QuestionDataset(split['question'], split['text'])

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

    def train(self, query_encoder, document_encoder, train_loader, val_loader, evaluator, epochs=20, lr=3e-4, warmup_ratio=0.05, monitor='mrr', patience=6, min_delta=1e-4, min_epochs=15):

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
        start_counting = 5
        best_val = float('inf')
        bad_epochs = 0

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
            print(f'Epoch {epoch + 1}/{epochs}: loss {total_loss / len(train_loader):.4f} | val loss (diagnostic) {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}')

            metrics = evaluator(query_encoder, document_encoder, device)
            print('  ' + ' | '.join(f'{name} {value:.4f}' for name, value in metrics.items()))
            score = val_loss

            if score < best_val - min_delta:
                best_val = score
                bad_epochs = 0
                best_state = (copy.deepcopy(query_encoder.state_dict()),
                                  copy.deepcopy(document_encoder.state_dict()))
            elif epoch + 1 > start_counting:
                bad_epochs += 1
                print(f'  no {monitor} improvement for {bad_epochs}/{patience} epochs (best {best_val:.4f})')
                if bad_epochs >= patience:
                    print(f'Early stopping megállt, best val {monitor}: {best_val:.4f}')
                    break

        print(f'Best val {monitor}: {best_val:.4f}')

        return query_encoder, document_encoder