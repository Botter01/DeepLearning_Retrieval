import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.loss import DPRLoss


def make_train_loader(split, tokenizer, batch_size=32, max_length=128):
    """
    @param split: corpus.train (HF Dataset with columns 'question' and 'text')
    @param tokenizer: BERT tokenizer, e.g. AutoTokenizer.from_pretrained('bert-base-uncased')

    Every batch is a dict of 4 tensors (B x L):
        query_ids, query_mask, document_ids, document_mask
    Row i of the queries matches row i of the documents.
    """
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

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module, train_loader, epochs=20, lr=0.0001):
        #raise NotImplementedError()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_encoder.to(device)
        document_encoder.to(device)

        optimizer = torch.optim.AdamW(list(query_encoder.parameters()) + list(document_encoder.parameters()), lr=lr)

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

                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs}: loss {total_loss / len(train_loader):.4f}')

        return query_encoder, document_encoder