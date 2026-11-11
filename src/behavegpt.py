from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass
class BehaveGPTConfiguration:
    num_activity_tokens: int
    activity_padding_index: int
    num_spatial_tokens: int
    spatial_padding_index: int
    num_temporal_tokens: int
    temporal_padding_index: int

    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: int = 0.1
    max_seq_len: int = 512


class BehaviourTokenizer(nn.Module):
    def __init__(self, config: BehaveGPTConfiguration):
        super().__init__()

        self.activity_embed = nn.Embedding(
            num_embeddings=config.num_activity_tokens,
            embedding_dim=config.d_model,
            padding_idx=config.activity_padding_index
        )
        self.spatial_embed = nn.Embedding(
            num_embeddings=config.num_spatial_tokens,
            embedding_dim=config.d_model,
        )
        self.temporal_embed = nn.Embedding(
            num_embeddings=config.num_temporal_tokens,
            embedding_dim=config.d_model
        )

        self.fusion = nn.Linear(config.d_model * 3, config.d_model)
        self.normalization = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        event_ids: torch.Tensor,
        spatial_ids: torch.Tensor,
        temporal_ids: torch.Tensor
    ) -> torch.Tensor:
        activity_embedded = self.activity_embed(event_ids)
        spatial_embedded = self.spatial_embed(spatial_ids)
        temporal_embedded = self.temporal_embed(temporal_ids) 

        merged = torch.concat(
            [activity_embedded, spatial_embedded, temporal_embedded],
            dim=-1
        )

        embeddings = self.fusion(merged)
        return self.dropout(self.normalization(embeddings))

class DRO(nn.Module):
    """
    Distributionally Robust Optimization (DRO) Loss to address long-tail user behaviors.
    Focuses loss weight dynamically on worst-performing groups/sub-distributions.
    """
    def __init__(self, num_groups: int, eta: float= 0.01):
        super().__init__()

        self.num_groups = num_groups
        self.eta = eta

        self.register_buffer("q", torch.ones(num_groups) / num_groups)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor
    ):
        sample_loss = functional.cross_entropy(logits, targets, reduction='none')

        group_losses = torch.zeros(self.num_groups, device=logits.device)
        for group in range(self.num_groups):
            mask = (group_ids == group)

            if mask.sum() > 0:
                group_losses[group] = sample_loss[mask].mean()

        if self.training:
            with torch.no_grad():
                self.q = self.q * torch.exp(self.eta * group_losses)
                self.q = self.q / self.q.sum()

        return torch.sum(self.q * group_losses)

class BehaveGPT(nn.Module):
    def __init__(self, config: BehaveGPTConfiguration):
        super().__init__()
        self.config = config
        self.tokenizer = BehaviourTokenizer(self.config)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
        )
        self.head_activity = nn.Linear(config.d_model, config.num_activity_tokens)
        self.__init_weight()

    def __init_weight(self):
        nn.init.normal_(self.pos_embed, std=0.02)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(
        self,
        activities: torch.Tensor,
        spatials: torch.Tensor,
        temporals: torch.Tensor,
        padding_mask: torch.BoolTensor = None
    ):
        _, seq_len = activities.shape

        x = self.tokenizer(activities, spatials, temporals)
        x = x + self.pos_embed[:, :seq_len, :]

        causal_mask = self.generate_causal_mask(seq_len, x.device)

        h = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )

        return self.head_activity(h)
