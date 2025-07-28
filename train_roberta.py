import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class RoBERTaConfig:
    vocab_size: int = 50265  # RoBERTa's vocab size
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514  # RoBERTa uses 512 + 2 for special tokens the {beginning,end}-of-sequence as <s> and </s>
    type_vocab_size: int = 1  # RoBERTa doesn't use token type embeddings
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2

class RoBERTaEmbeddings(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id)
        # Add token type embeddings to match HuggingFace (even though RoBERTa doesn't use them)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        # input_ids: (B, T)
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            # position_ids = self.position_ids[:, :seq_length]
            from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
            position_ids = create_position_ids_from_input_ids(
                input_ids, self.word_embeddings.padding_idx, past_key_values_length=0
            )
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        embeddings = self.word_embeddings(input_ids)  # (B, T, C)
        position_embeddings = self.position_embeddings(position_ids)  # (B, T, C)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)  # (B, T, C)
        
        # embeddings = embeddings + position_embeddings + token_type_embeddings  # (B, T, C)
        embeddings = embeddings + position_embeddings  # (B, T, C)
        embeddings = self.LayerNorm(embeddings)  # (B, T, C)
        embeddings = self.dropout(embeddings)  # (B, T, C)
        return embeddings  # (B, T, C)

class RoBERTaSelfAttention(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_attention_heads")
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # x: (B, T, C)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # (B, T, nh, hs)
        return x.permute(0, 2, 1, 3)  # (B, nh, T, hs)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (B, T, C)
        query_layer = self.transpose_for_scores(self.query(hidden_states))  # (B, nh, T, hs)
        key_layer = self.transpose_for_scores(self.key(hidden_states))      # (B, nh, T, hs)
        value_layer = self.transpose_for_scores(self.value(hidden_states))  # (B, nh, T, hs)

        # attention_scores: (B, nh, T, T)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # attention_mask: (B, 1, 1, T)
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)  # (B, nh, T, T)
        attention_probs = self.dropout(attention_probs)  # (B, nh, T, T)

        context_layer = torch.matmul(attention_probs, value_layer)  # (B, nh, T, hs)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (B, T, nh, hs)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (B, T, C)

        return context_layer, attention_probs  # (B, T, C), (B, nh, T, T)

class RoBERTaSelfOutput(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states: (B, T, C)
        # input_tensor: (B, T, C)
        hidden_states = self.dense(hidden_states)  # (B, T, C)
        hidden_states = self.dropout(hidden_states)  # (B, T, C)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # (B, T, C)
        return hidden_states  # (B, T, C)

class RoBERTaAttention(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.self = RoBERTaSelfAttention(config)
        self.output = RoBERTaSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (B, T, C)
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)  # (B, T, C)
        return attention_output, self_outputs[1]  # (B, T, C), (B, nh, T, T)

class RoBERTaIntermediate(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states):
        # hidden_states: (B, T, C)
        hidden_states = self.dense(hidden_states)  # (B, T, C_ff)
        hidden_states = self.intermediate_act_fn(hidden_states)  # (B, T, C_ff)
        return hidden_states  # (B, T, C_ff)

class RoBERTaOutput(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states: (B, T, C_ff)
        # input_tensor: (B, T, C)
        hidden_states = self.dense(hidden_states)  # (B, T, C)
        hidden_states = self.dropout(hidden_states)  # (B, T, C)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # (B, T, C)
        return hidden_states  # (B, T, C)

class RoBERTaLayer(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.attention = RoBERTaAttention(config)
        self.intermediate = RoBERTaIntermediate(config)
        self.output = RoBERTaOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (B, T, C)
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)  # (B, T, C), (B, nh, T, T)
        intermediate_output = self.intermediate(attention_output)  # (B, T, C_ff)
        layer_output = self.output(intermediate_output, attention_output)  # (B, T, C)
        return layer_output, attention_probs  # (B, T, C), (B, nh, T, T)

class RoBERTaEncoder(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.layer = nn.ModuleList([RoBERTaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (B, T, C)
        all_hidden_states = []
        all_attentions = []
        
        for i, layer_module in enumerate(self.layer):
            all_hidden_states.append(hidden_states)  # (B, T, C)
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)  # (B, T, C), (B, nh, T, T)
            all_attentions.append(attention_probs)
        
        all_hidden_states.append(hidden_states)  # (B, T, C)
        
        return hidden_states, all_hidden_states, all_attentions  # (B, T, C), list[(B, T, C)], list[(B, nh, T, T)]

class RoBERTaPooler(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states: (B, T, C)
        first_token_tensor = hidden_states[:, 0]  # (B, C)
        pooled_output = self.dense(first_token_tensor)  # (B, C)
        pooled_output = self.activation(pooled_output)  # (B, C)
        return pooled_output  # (B, C)

class RoBERTaLMHead(nn.Module):
    """RoBERTa Head for masked language modeling."""
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)  # Include bias
        # Additional bias parameter to match HuggingFace exactly
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features):
        # features: (B, T, C)
        x = self.dense(features)  # (B, T, C)
        x = F.gelu(x)  # (B, T, C)
        x = self.layer_norm(x)  # (B, T, C)
        x = self.decoder(x)  # (B, T, vocab_size) - includes decoder.bias
        x = x + self.bias  # Add the additional bias to match HuggingFace
        return x  # (B, T, vocab_size)

class RoBERTaModel(nn.Module):
    def __init__(self, config: RoBERTaConfig, use_pooler=True):
        super().__init__()
        self.config = config
        self.embeddings = RoBERTaEmbeddings(config)
        self.encoder = RoBERTaEncoder(config)
        self.pooler = RoBERTaPooler(config) if use_pooler else None
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_extended_attention_mask(self, attention_mask, input_shape):
        # attention_mask: (B, T)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]  # (B, 1, T, T)
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask  # (B, 1, 1, T) or (B, 1, T, T)

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, use_pooler=True):
        """
        Load from pretrained HuggingFace RoBERTa model
        
        Args:
            pretrained_model_or_path: Either:
                - A string, the model id of a pretrained model hosted on huggingface.co,
                - A path to a directory containing model weights
            use_pooler: Whether to use the pooler in the model
                
        Returns:
            A RoBERTaModel instance with loaded weights
        """
        try:
            from transformers import RobertaModel, RobertaConfig as HfRobertaConfig
        except ImportError:
            raise ImportError("You need to install transformers to use from_pretrained()")
            
        print(f"Loading pretrained model from {pretrained_model_or_path}")
        
        # Load HuggingFace model and config
        hf_config = HfRobertaConfig.from_pretrained(pretrained_model_or_path)
        hf_model = RobertaModel.from_pretrained(pretrained_model_or_path)
        
        # Create our config from HF config
        config = RoBERTaConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=hf_config.hidden_act,
            hidden_dropout_prob=hf_config.hidden_dropout_prob,
            attention_probs_dropout_prob=hf_config.attention_probs_dropout_prob,
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=hf_config.type_vocab_size,
            initializer_range=hf_config.initializer_range,
            layer_norm_eps=hf_config.layer_norm_eps,
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
        )
        
        # Create our model with this config
        model = cls(config, use_pooler=use_pooler)
        
        # Get HF state dict
        hf_state_dict = hf_model.state_dict()
        
        # Create a mapping between HF and our model
        key_mapping = {
            # Embeddings
            "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
            # "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
            "embeddings.LayerNorm.weight": "embeddings.LayerNorm.weight",
            "embeddings.LayerNorm.bias": "embeddings.LayerNorm.bias",
            
            # Encoder layers - we'll handle these in a loop
        }
        
        # Add encoder layer mappings
        for i in range(config.num_hidden_layers):
            # Attention weights
            key_mapping.update({
                f"encoder.layer.{i}.attention.self.query.weight": f"encoder.layer.{i}.attention.self.query.weight",
                f"encoder.layer.{i}.attention.self.query.bias": f"encoder.layer.{i}.attention.self.query.bias",
                f"encoder.layer.{i}.attention.self.key.weight": f"encoder.layer.{i}.attention.self.key.weight",
                f"encoder.layer.{i}.attention.self.key.bias": f"encoder.layer.{i}.attention.self.key.bias",
                f"encoder.layer.{i}.attention.self.value.weight": f"encoder.layer.{i}.attention.self.value.weight",
                f"encoder.layer.{i}.attention.self.value.bias": f"encoder.layer.{i}.attention.self.value.bias",
                f"encoder.layer.{i}.attention.output.dense.weight": f"encoder.layer.{i}.attention.output.dense.weight",
                f"encoder.layer.{i}.attention.output.dense.bias": f"encoder.layer.{i}.attention.output.dense.bias",
                f"encoder.layer.{i}.attention.output.LayerNorm.weight": f"encoder.layer.{i}.attention.output.LayerNorm.weight",
                f"encoder.layer.{i}.attention.output.LayerNorm.bias": f"encoder.layer.{i}.attention.output.LayerNorm.bias",
                
                # FFN weights
                f"encoder.layer.{i}.intermediate.dense.weight": f"encoder.layer.{i}.intermediate.dense.weight",
                f"encoder.layer.{i}.intermediate.dense.bias": f"encoder.layer.{i}.intermediate.dense.bias",
                f"encoder.layer.{i}.output.dense.weight": f"encoder.layer.{i}.output.dense.weight",
                f"encoder.layer.{i}.output.dense.bias": f"encoder.layer.{i}.output.dense.bias",
                f"encoder.layer.{i}.output.LayerNorm.weight": f"encoder.layer.{i}.output.LayerNorm.weight",
                f"encoder.layer.{i}.output.LayerNorm.bias": f"encoder.layer.{i}.output.LayerNorm.bias",
            })
        
        # Add pooler weights if needed
        if use_pooler:
            key_mapping.update({
                "pooler.dense.weight": "pooler.dense.weight",
                "pooler.dense.bias": "pooler.dense.bias",
            })
        
        # Create our state dict from HF weights
        state_dict = {}
        for hf_key, our_key in key_mapping.items():
            hf_full_key = f"roberta.{hf_key}"
            if hf_full_key in hf_state_dict:
                state_dict[our_key] = hf_state_dict[hf_full_key].clone()
            else:
                print(f"Warning: key {hf_full_key} not found in HF model")
        
        # Load weights into our model
        model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded {len(state_dict)} weights from HuggingFace model")
        return model

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, T)
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)  # (B, T)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)  # (B, 1, 1, T)

        embedding_output = self.embeddings(input_ids)  # (B, T, C)
        encoder_output, all_hidden_states, all_attentions = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None  # (B, C) or None

        return encoder_output, pooled_output, all_hidden_states, all_attentions  # (B, T, C), (B, C) or None, ...

class RoBERTaForMaskedLM(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.roberta = RoBERTaModel(config, use_pooler=False)
        self.lm_head = RoBERTaLMHead(config)
        
        # Tie weights to match HuggingFace exactly
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path):
        """
        Load from pretrained HuggingFace RoBERTaForMaskedLM model
        
        Args:
            pretrained_model_or_path: Either:
                - A string, the model id of a pretrained model hosted on huggingface.co,
                - A path to a directory containing model weights
                
        Returns:
            A RoBERTaForMaskedLM instance with loaded weights
        """
        try:
            from transformers import RobertaForMaskedLM, RobertaConfig as HfRobertaConfig
        except ImportError:
            raise ImportError("You need to install transformers to use from_pretrained()")
            
        print(f"Loading pretrained MLM model from {pretrained_model_or_path}")
        
        # Load HuggingFace model and config
        hf_config = HfRobertaConfig.from_pretrained(pretrained_model_or_path)
        hf_model = RobertaForMaskedLM.from_pretrained(pretrained_model_or_path, attn_implementation="eager")
        
        # Create our config from HF config
        config = RoBERTaConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=hf_config.hidden_act,
            hidden_dropout_prob=hf_config.hidden_dropout_prob,
            attention_probs_dropout_prob=hf_config.attention_probs_dropout_prob,
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=hf_config.type_vocab_size,
            initializer_range=hf_config.initializer_range,
            layer_norm_eps=hf_config.layer_norm_eps,
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
        )
        
        # Create our model with this config
        model = cls(config)
        
        # Get both state dicts
        hf_state_dict = hf_model.state_dict()
        
        # Create complete mapping between HF and our model
        key_mapping = {}
        
        # RoBERTa backbone mappings  
        # Embeddings
        key_mapping.update({
            "roberta.embeddings.word_embeddings.weight": "roberta.embeddings.word_embeddings.weight",
            "roberta.embeddings.position_embeddings.weight": "roberta.embeddings.position_embeddings.weight",
            # "roberta.embeddings.token_type_embeddings.weight": "roberta.embeddings.token_type_embeddings.weight",
            "roberta.embeddings.LayerNorm.weight": "roberta.embeddings.LayerNorm.weight",
            "roberta.embeddings.LayerNorm.bias": "roberta.embeddings.LayerNorm.bias",
        })
        
        # Encoder layers
        for i in range(config.num_hidden_layers):
            key_mapping.update({
                # Attention
                f"roberta.encoder.layer.{i}.attention.self.query.weight": f"roberta.encoder.layer.{i}.attention.self.query.weight",
                f"roberta.encoder.layer.{i}.attention.self.query.bias": f"roberta.encoder.layer.{i}.attention.self.query.bias",
                f"roberta.encoder.layer.{i}.attention.self.key.weight": f"roberta.encoder.layer.{i}.attention.self.key.weight",
                f"roberta.encoder.layer.{i}.attention.self.key.bias": f"roberta.encoder.layer.{i}.attention.self.key.bias",
                f"roberta.encoder.layer.{i}.attention.self.value.weight": f"roberta.encoder.layer.{i}.attention.self.value.weight",
                f"roberta.encoder.layer.{i}.attention.self.value.bias": f"roberta.encoder.layer.{i}.attention.self.value.bias",
                f"roberta.encoder.layer.{i}.attention.output.dense.weight": f"roberta.encoder.layer.{i}.attention.output.dense.weight",
                f"roberta.encoder.layer.{i}.attention.output.dense.bias": f"roberta.encoder.layer.{i}.attention.output.dense.bias",
                f"roberta.encoder.layer.{i}.attention.output.LayerNorm.weight": f"roberta.encoder.layer.{i}.attention.output.LayerNorm.weight",
                f"roberta.encoder.layer.{i}.attention.output.LayerNorm.bias": f"roberta.encoder.layer.{i}.attention.output.LayerNorm.bias",
                
                # FFN
                f"roberta.encoder.layer.{i}.intermediate.dense.weight": f"roberta.encoder.layer.{i}.intermediate.dense.weight",
                f"roberta.encoder.layer.{i}.intermediate.dense.bias": f"roberta.encoder.layer.{i}.intermediate.dense.bias",
                f"roberta.encoder.layer.{i}.output.dense.weight": f"roberta.encoder.layer.{i}.output.dense.weight",
                f"roberta.encoder.layer.{i}.output.dense.bias": f"roberta.encoder.layer.{i}.output.dense.bias",
                f"roberta.encoder.layer.{i}.output.LayerNorm.weight": f"roberta.encoder.layer.{i}.output.LayerNorm.weight",
                f"roberta.encoder.layer.{i}.output.LayerNorm.bias": f"roberta.encoder.layer.{i}.output.LayerNorm.bias",
            })
        
        # LM head mappings
        key_mapping.update({
            "lm_head.dense.weight": "lm_head.dense.weight",
            "lm_head.dense.bias": "lm_head.dense.bias",
            "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
            "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
            "lm_head.decoder.weight": "lm_head.decoder.weight",  # This will be tied to embeddings
            "lm_head.decoder.bias": "lm_head.decoder.bias",      # Decoder bias
            "lm_head.bias": "lm_head.bias",                      # Additional bias
        })
        
        # Load weights into our model
        our_state_dict = {}
        missing_keys = []
        unexpected_keys = []
        
        for hf_key, our_key in key_mapping.items():
            if hf_key in hf_state_dict:
                our_state_dict[our_key] = hf_state_dict[hf_key].clone()
            else:
                missing_keys.append(hf_key)
                
        # Load the weights
        load_result = model.load_state_dict(our_state_dict, strict=False)
        missing_keys.extend(load_result.missing_keys)
        unexpected_keys.extend(load_result.unexpected_keys)
        
        # Report results
        print(f"Successfully loaded {len(our_state_dict)} weights from HuggingFace model")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        # Verify parameter counts match
        hf_params = sum(p.numel() for p in hf_model.parameters())
        our_params = sum(p.numel() for p in model.parameters())
        print(f"HuggingFace model parameters: {hf_params:,}")
        print(f"Our model parameters: {our_params:,}")
        print(f"Parameter count match: {'✓' if hf_params == our_params else '✗'}")
        
        return model

    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids: (B, T)
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (B, T, C)
        prediction_scores = self.lm_head(sequence_output)  # (B, T, vocab_size)

        masked_lm_loss = None
        if labels is not None:
            # labels: (B, T)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))

        return masked_lm_loss, prediction_scores, outputs[2:]  # (scalar), (B, T, vocab_size), ...

# MLM Data preprocessing utilities
class MLMDataProcessor:
    """Data processor for Masked Language Modeling following RoBERTa strategy"""
    
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        # RoBERTa special tokens
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', 50264)  # <mask>
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 1)        # <pad>
        self.cls_token_id = getattr(tokenizer, 'cls_token_id', 0)        # <s>
        self.sep_token_id = getattr(tokenizer, 'sep_token_id', 2)        # </s>
        
    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        for special_token_id in [self.pad_token_id, self.cls_token_id, self.sep_token_id]:
            special_tokens_mask |= (labels == special_token_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else range(50265)), 
                                   labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

# Simple tokenizer class for compatibility
class SimpleTokenizer:
    """Simple tokenizer wrapper for testing"""
    def __init__(self, vocab_size=50265):
        self.vocab_size = vocab_size
        self.mask_token_id = 50264  # <mask>
        self.pad_token_id = 1       # <pad>
        self.cls_token_id = 0       # <s>
        self.sep_token_id = 2       # </s>
    
    def get_vocab(self):
        return {f"token_{i}": i for i in range(self.vocab_size)}

# Training utilities following llm.c patterns
def configure_optimizers(model, weight_decay, learning_rate, betas=(0.9, 0.999)):
    """Configure optimizer similar to llm.c train_gpt2.py"""
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

def get_batch(data, batch_size, block_size, device='cpu'):
    """Generate a batch of data following llm.c patterns"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    return x.to(device)

def estimate_loss(model, data, eval_iters=100, batch_size=4, block_size=128, device='cpu'):
    """Estimate loss on data following llm.c patterns"""
    model.eval()
    losses = torch.zeros(eval_iters)
    mlm_processor = MLMDataProcessor(SimpleTokenizer())
    
    for k in range(eval_iters):
        X = get_batch(data, batch_size, block_size, device)
        attention_mask = (X != 1).long()  # 1 is pad token
        
        # Apply MLM masking
        X_masked, labels = mlm_processor.mask_tokens(X)
        
        with torch.no_grad():
            loss, logits, _ = model(X_masked, attention_mask=attention_mask, labels=labels)
            losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    text = "The capital of France is <mask>."
    inputs = tokenizer(text, return_tensors="pt")

    # model = RoBERTaForMaskedLM.from_pretrained("roberta-base")
    model = RoBERTaForMaskedLM(RoBERTaConfig())
    torch.manual_seed(42)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        logits = model(**inputs)
        logits = logits[1]
    print(logits.shape)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    print(f"Predicted token: {predicted_token}")
    top_k = 10
    top_k_logits, top_k_indices = torch.topk(logits[0, mask_token_index], top_k)
    for i in range(top_k):
        token_id = top_k_indices.view(-1)[i].item()
        token = tokenizer.decode(token_id)
        score = torch.softmax(top_k_logits.view(-1)[i], dim=0).item()
        print(f"Token: {token}, Score: {score:.4f}")
    # # Test basic model
    # print("="*50)
    # print("Testing RoBERTa Implementation")
    # print("="*50)
    
    # config = RoBERTaConfig()
    # model = RoBERTaForMaskedLM(config)
    
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # print(f"Model config: {config}")
    
    # # Test forward pass
    # batch_size = 2
    # seq_length = 128
    # input_ids = torch.randint(3, config.vocab_size-1, (batch_size, seq_length))  # Avoid special tokens
    # attention_mask = torch.ones(batch_size, seq_length)
    
    # # Test MLM preprocessing
    # print("\nTesting MLM preprocessing:")
    # tokenizer = SimpleTokenizer()
    # mlm_processor = MLMDataProcessor(tokenizer)
    
    # input_ids_masked, labels = mlm_processor.mask_tokens(input_ids.clone())
    
    # print(f"Original tokens (first 20): {input_ids[0][:20].tolist()}")
    # print(f"Masked tokens (first 20):   {input_ids_masked[0][:20].tolist()}")
    # print(f"Labels (first 20):         {labels[0][:20].tolist()}")
    
    # num_masked = (input_ids_masked == tokenizer.mask_token_id).sum().item()
    # print(f"Number of masked tokens: {num_masked} ({num_masked/(batch_size*seq_length)*100:.1f}%)")
    
    # # Test model forward pass with MLM
    # print("\nTesting model forward pass:")
    # with torch.no_grad():
    #     loss, logits, _ = model(input_ids_masked, attention_mask=attention_mask, labels=labels)
    #     print(f"MLM Loss: {loss.item():.4f}")
    #     print(f"Logits shape: {logits.shape}")
    #     print(f"Expected shape: ({batch_size}, {seq_length}, {config.vocab_size})")
    
    # # Test optimizer configuration
    # print("\nTesting optimizer configuration:")
    # optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=1e-4)
    # print(f"Optimizer: {optimizer}")
    
    # print("\n" + "="*50)
    # print("RoBERTa implementation test completed successfully!")
    # print("="*50)
    
    # # Next steps guide
    # print("\nNext steps to complete the implementation:")
    # print("1. Test against HuggingFace RoBERTa for parity")
    # print("2. Add data loading for real datasets") 
    # print("3. Implement training loop")
    # print("4. Add model saving/loading compatible with llm.c format")
    # print("5. Test with pretrained weights loading")
