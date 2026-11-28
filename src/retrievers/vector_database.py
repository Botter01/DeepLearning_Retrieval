import os, json
import torch
import faiss
from transformers import AutoTokenizer
from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.retrievers.models.encoder import TransformerEncoder
from src.evaluation.metrics import Evaluate


class VectorDatabase:
    def __init__(self, embeddings, ids):
        self.ids = ids
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(self._prep(embeddings))

    @staticmethod
    def _prep(x):
        x = x.numpy().astype('float32')
        faiss.normalize_L2(x)
        return x

    def search(self, query_embeddings, k):
        doc, idx = self.index.search(self._prep(query_embeddings), k)
        return [[self.ids[j] for j in row] for row in idx]


@torch.no_grad()
def encode(texts, encoder, tokenizer, device, batch = 256, max_length = 512):
    out = []
    for i in range(0, len(texts), batch):
        t = tokenizer(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_length, return_tensors='pt').to(device)
        out.append(encoder(t['input_ids'], t['attention_mask']).cpu())
    return torch.cat(out)


def flatten_documents(corpus):
    texts, ids = [], []
    for row_texts, row_ids in zip(corpus.all_documents['text'], corpus.all_documents['document_id']):
        texts.extend(row_texts)
        ids.extend(row_ids)
    return texts, ids


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints'
    results_dir = 'results'
    k = 6  

    corpus = Corpus(dataset_name='fiqa', base_dir='./retrieval_data')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    query_encoder = TransformerEncoder(n_layers=5, n_heads=2, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='mean')
    document_encoder = TransformerEncoder(n_layers=5, n_heads=2, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='mean')
    query_encoder.load_state_dict(torch.load(f'{save_dir}/query_encoder.pt', map_location=device))
    document_encoder.load_state_dict(torch.load(f'{save_dir}/document_encoder.pt', map_location=device))
    query_encoder.to(device).eval()
    document_encoder.to(device).eval()

    doc_texts, doc_ids = flatten_documents(corpus)
    print(f'Encoding {len(doc_texts)} documents...')
    db = VectorDatabase(encode(doc_texts, document_encoder, tokenizer, device), doc_ids)

    def run(split):
        queries = [Query(split['question'][i], id=split['query_id'][i],
                         relevant_document_ids=corpus.relevant_ids[split['query_id'][i]])
                   for i in range(split.num_rows)]
        q_emb = encode([q.getQuery() for q in queries], query_encoder, tokenizer, device)
        retrieved = db.search(q_emb, k)
        rels = [[doc_id in set(q.relevant_document_ids) for doc_id in row]
                for q, row in zip(queries, retrieved)]
        return Evaluate(rels, queries, k=k)

    all_scores = {name: run(split) for name, split in
                  [('train', corpus.train), ('val', corpus.val), ('test', corpus.test)]}
    for name, scores in all_scores.items():
        print(f'{name}: {scores}')

    os.makedirs(results_dir, exist_ok=True)
    with open(f'{results_dir}/metrics_DENSE_4.json', 'w') as f:
        json.dump(all_scores, f, indent=2)
