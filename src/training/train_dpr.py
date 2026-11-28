import os
import torch
from transformers import AutoTokenizer
from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
from src.data_utils.query import Query
from src.retrievers.vector_database import flatten_documents
from src.training.trainer import Trainer, make_train_loader, RetrievalEvaluator

batch_size = 64
epochs = 100
lr = 3e-4 
save_dir = 'checkpoints'

corpus = Corpus(dataset_name='fiqa', base_dir='./retrieval_data')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_loader = make_train_loader(corpus.train, tokenizer, batch_size=batch_size)
val_loader = make_train_loader(corpus.val, tokenizer, batch_size=batch_size)

query_encoder = TransformerEncoder(n_layers=5, n_heads=2, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='mean')
document_encoder = TransformerEncoder(n_layers=5, n_heads=2, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=tokenizer.vocab_size, pooling_type='mean')

doc_texts, doc_ids = flatten_documents(corpus)
val_queries = [Query(corpus.val['question'][i], id=corpus.val['query_id'][i],
                     relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][i]])
               for i in range(corpus.val.num_rows)]
evaluator = RetrievalEvaluator(doc_texts, doc_ids, val_queries, tokenizer, k=10)

trainer = Trainer()

query_encoder, document_encoder = trainer.train(query_encoder, document_encoder, train_loader, val_loader, evaluator, epochs=epochs, lr=lr)

os.makedirs(save_dir, exist_ok=True)
torch.save(query_encoder.state_dict(), f'{save_dir}/query_encoder.pt')
torch.save(document_encoder.state_dict(), f'{save_dir}/document_encoder.pt')
print(f'Saved encoders to {save_dir}/')