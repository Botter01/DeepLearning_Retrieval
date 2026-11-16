"""
Train the DPR query and document encoders on FiQA.
Run from the repo root:  python train_dpr.py

Before the first run, download the tokenizer once (corpus.py switches Hugging Face to offline mode):
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"
"""
import os
import torch
from transformers import AutoTokenizer
from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
from src.training.trainer import Trainer, make_train_loader

# ---- settings (same model sizes as dpr.py / config/model/encoder.yaml) ----
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
SAVE_DIR = 'checkpoints'

# 1) data: the same Corpus that main.py uses
corpus = Corpus(dataset_name='fiqa', base_dir='./retrieval_data')

# 2) tokenizer: raw text -> token ids + attention mask
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_loader = make_train_loader(corpus.train, tokenizer, batch_size=BATCH_SIZE)

# 3) two encoders. vocab_size MUST be the tokenizer's (30522 for BERT)
def make_encoder():
    return TransformerEncoder(n_layers=6, n_heads=4, emb_dim=128, hidden_dim=32, feed_forward_dim=768,
                              vocab_size=tokenizer.vocab_size, pooling_type='cls')

query_encoder = make_encoder()
document_encoder = make_encoder()

# 4) train
query_encoder, document_encoder = Trainer().train(query_encoder, document_encoder, train_loader,
                                                  epochs=EPOCHS, lr=LR)

# 5) save the weights (dpr.py will load these later)
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(query_encoder.state_dict(), f'{SAVE_DIR}/query_encoder.pt')
torch.save(document_encoder.state_dict(), f'{SAVE_DIR}/document_encoder.pt')
print(f'Saved encoders to {SAVE_DIR}/')