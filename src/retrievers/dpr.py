from src.retrievers.models.encoder import TransformerEncoder

class DPR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__load_models()
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        query_encoder = TransformerEncoder(n_heads=4, n_layers=6, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=30000)
        document_encoder = TransformerEncoder(n_heads=4, n_layers=6, emb_dim=128, hidden_dim=32, feed_forward_dim=768, vocab_size=30000)
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('document_encoder_weights_path', None):
            document_encoder.load_state_dict(document_weights_path)

    

        

