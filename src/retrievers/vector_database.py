import os, json
import torch
from transformers import AutoTokenizer
from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.retrievers.models.encoder import TransformerEncoder
from src.evaluation.metrics import Evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = 'checkpoints'
results_dir= 'results'
k = 6

corpus = Corpus(dataset_name='fiqa', base_dir='./retrieval_data')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

query_encoder = TransformerEncoder(n_layers=3, n_heads=4, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='mean')
document_encoder = TransformerEncoder(n_layers=3, n_heads=4, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='mean')

query_encoder.load_state_dict(torch.load(f'{save_dir}/query_encoder.pt', map_location=device))
document_encoder.load_state_dict(torch.load(f'{save_dir}/document_encoder.pt', map_location=device))
query_encoder.to(device).eval()
document_encoder.to(device).eval()

def make_queries(split):
    return [Query(split['question'][i], id=split['query_id'][i],
                  relevant_document_ids=corpus.relevant_ids[split['query_id'][i]])
            for i in range(split.num_rows)]

val_queries = make_queries(corpus.val)
test_queries = make_queries(corpus.test)

@torch.no_grad()
def encode(texts, encoder, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        t = tokenizer(texts[i:i+bs], padding=True, truncation=True,
                      max_length=128, return_tensors='pt').to(device)
        out.append(encoder(t['input_ids'], t['attention_mask']).cpu())
    return torch.cat(out)

doc_ids = [d[0] for d in corpus.all_documents['document_id']]
doc_texts = [' '.join(t) for t in corpus.all_documents['text']]

doc_matrix = encode(doc_texts, document_encoder)

def run(split, queries):
    q = encode(list(split['question']), query_encoder)
    top = (q @ doc_matrix.T).topk(k, dim=1).indices.tolist()
    rels = [[doc_ids[j] in set(corpus.relevant_ids[qr.id]) for j in row]
            for qr, row in zip(queries, top)]
    return Evaluate(rels, queries, k=k)

val_scores = run(corpus.val, val_queries)
test_scores = run(corpus.test, test_queries)
print('Val: ', val_scores)
print('Test:', test_scores)

os.makedirs(results_dir, exist_ok=True)
with open(f'{results_dir}/metrics_DENSE.json', 'w') as f:
    json.dump({'val': val_scores, 'test': test_scores}, f, indent=2)