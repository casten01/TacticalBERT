import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForMaskedLM
from .config import Config

class TacticalEmbeddings(nn.Module):
    """
    Adapted Embeddings class of HuggingFace's BERT to include additional embeddings for location, duration, and context features.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.loc_embeddings = nn.Embedding(Config.GRID_WIDTH * Config.GRID_HEIGHT + 100, config.hidden_size, padding_idx=0)
        self.duration_embeddings = nn.Embedding(Config.DURATION_BINS_NUMBER, config.hidden_size, padding_idx=0)
        self.context_projection = nn.Linear(3, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, 
                loc_ids=None, duration_ids=None, context_features=None, **kwargs):
        
        if inputs_embeds is None:
            words_embeddings = self.word_embeddings(input_ids)
        else:
            words_embeddings = inputs_embeds
            
        if input_ids is not None:
            seq_length = input_ids.size(1)
        else:
            seq_length = inputs_embeds.size(1)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=words_embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand_as(words_embeddings[:, :, 0]) 

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (words_embeddings.size(0), seq_length), 
                dtype=torch.long, 
                device=words_embeddings.device
            )

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        
        if loc_ids is not None:
            embeddings += self.loc_embeddings(loc_ids)
        if duration_ids is not None:
            embeddings += self.duration_embeddings(duration_ids)
        if context_features is not None:
            embeddings += self.context_projection(context_features)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TacticalBert(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_hidden_layers=Config.LAYERS,
            num_attention_heads=Config.HEADS,
            intermediate_size=Config.HIDDEN_SIZE * 4,
            max_position_embeddings=Config.MAX_LEN,
            pad_token_id=0
        )
        
        self.bert = BertForMaskedLM(self.config)
        
        self.bert.bert.embeddings = TacticalEmbeddings(self.config)

    def forward(self, input_ids, attention_mask, labels=None, loc_ids=None, duration_ids=None, context_features=None, output_attentions=False):
        
        embedding_output = self.bert.bert.embeddings(
            input_ids, 
            loc_ids=loc_ids, 
            duration_ids=duration_ids, 
            context_features=context_features
        )
        
        outputs = self.bert(
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions
        )
        
        return outputs