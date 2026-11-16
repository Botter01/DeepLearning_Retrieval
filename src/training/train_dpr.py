import os
import torch
from transformers import AutoTokenizer
from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
from src.training.trainer import Trainer, make_train_loader

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
SAVE_DIR = 'checkpoints'

corpus = Corpus(dataset_name='fiqa', base_dir='./retrieval_data')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_loader = make_train_loader(corpus.train, tokenizer, batch_size=BATCH_SIZE)

query_encoder = TransformerEncoder(n_layers=6, n_heads=4, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='cls')
document_encoder = TransformerEncoder(n_layers=6, n_heads=4, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='cls')


query_encoder, document_encoder = Trainer().train(query_encoder, document_encoder, train_loader, epochs=EPOCHS, lr=LR)

os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(query_encoder.state_dict(), f'{SAVE_DIR}/query_encoder.pt')
torch.save(document_encoder.state_dict(), f'{SAVE_DIR}/document_encoder.pt')
print(f'Saved encoders to {SAVE_DIR}/')