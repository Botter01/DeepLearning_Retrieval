import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.vector_database import flatten_documents

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
corpus = Corpus(dataset_name='fiqa', base_dir='./retrieval_data')

@torch.no_grad()
def plot_attention(enc, text, name):
    t = tokenizer(text, truncation=True, max_length=32, return_tensors='pt').to(device)
    enc(t['input_ids'], t['attention_mask'])
    tokens = tokenizer.convert_ids_to_tokens(t['input_ids'][0])
    n_layers, n_heads = len(enc.model), enc.n_heads

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers), squeeze=False)
    for l, layer in enumerate(enc.model):
        attn = layer.self_attn.attn[0].cpu()
        for h in range(n_heads):
            ax = axes[l][h]
            ax.imshow(attn[h], cmap='viridis')
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens)
            ax.set_title(f'layer {l} head {h}')
    plt.tight_layout()
    plt.savefig(f'attn_{name}.png')
    plt.show()


query_encoder = TransformerEncoder(n_layers=5, n_heads=2, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='mean')
query_encoder.load_state_dict(torch.load('checkpoints/query_encoder.pt', map_location=device))
query_encoder = query_encoder.to(device).eval()

document_encoder = TransformerEncoder(n_layers=5, n_heads=2, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='mean')
document_encoder.load_state_dict(torch.load('checkpoints/document_encoder.pt', map_location=device))
document_encoder = document_encoder.to(device).eval()

i = 0
query_text = corpus.test['question'][i]
rel_id = corpus.relevant_ids[corpus.test['query_id'][i]][0]
doc_texts, doc_ids = flatten_documents(corpus)
doc_text = doc_texts[doc_ids.index(rel_id)]

plot_attention(query_encoder, query_text, 'query')
plot_attention(document_encoder, doc_text, 'doc')