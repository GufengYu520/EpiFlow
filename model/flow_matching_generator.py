import torch
import torch.nn as nn
from torch import Tensor
import esm
from esm.modules import RobertaLMHead


# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        half_dim = time_dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.freqs = torch.exp(-embeddings * torch.arange(half_dim))
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            Swish(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
    def forward(self, t):
        self.freqs = self.freqs.to(t.device)
        t_expanded = t.unsqueeze(-1) * self.freqs
        embeddings = torch.cat([torch.sin(t_expanded), torch.cos(t_expanded)], dim=-1)
        return self.mlp(embeddings)


class ConditionalESMBlock(nn.Module):
    def __init__(self, esm_block, hidden_dim, condition_dim):
        super().__init__()
        self.block = esm_block
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_dim * 2)
        )

    def forward(self, x, condition_embedding, self_attn_padding_mask=None, need_head_weights=False):
        gamma, beta = self.modulation(condition_embedding).chunk(2, dim=-1)
        
        gamma = gamma.transpose(0, 1)
        beta = beta.transpose(0, 1)

        x = self.layer_norm(x) * (1 + gamma) + beta
        
        output = self.block(x, self_attn_padding_mask=self_attn_padding_mask, need_head_weights=need_head_weights)
        return output



class ESM(nn.Module):
    def __init__(
            self, 
            esm_model, 
            # input_dim: int = 24, 
            time_dim: int = 128, 
            input_esm_dim: int = 320, 
            # pro_dim: int = 128,
            # hidden_dim: int = 128,
            length: int = 17,
            condition: bool = True,
            flow_matching_type: str = 'discrete',
            adaptive: bool = False):
        super().__init__()
        if esm_model == 'esm2_8m':
            self.esm_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
        elif esm_model == 'esm2_150m':
            self.esm_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
        self.esm_model.train()

        self.length = length
        self.adaptive = adaptive

        # Time embedding
        self.time_embedding = TimeEmbedding(input_esm_dim)

        if condition:
            # self.pro_embedding = nn.Linear(input_esm_dim, pro_dim)
            self.mhc_mlp = nn.Sequential(
                nn.Linear(input_esm_dim, input_esm_dim),
                nn.SiLU(),
                nn.Linear(input_esm_dim, input_esm_dim)
            )

        if adaptive:
            self.condition_mlp = nn.Sequential(
                nn.Linear(input_esm_dim*2, input_esm_dim*2),
                nn.SiLU(),
                nn.Linear(input_esm_dim*2, input_esm_dim)
            )

            self.conditional_layers = nn.ModuleList([
                ConditionalESMBlock(
                    esm_block=layer,
                    hidden_dim=input_esm_dim,
                    condition_dim=input_esm_dim 
                ) for layer in self.esm_model.layers
            ])


    def esm2_forward(self, x, repr_layers=[], padding_mask=None, need_head_weights=False, return_contacts=False):
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        # if not padding_mask.any():
        #     padding_mask = None

        for layer_idx, layer in enumerate(self.esm_model.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.esm_model.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.esm_model.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        return result
    

    def esm2_adaptive_forward(self, x, t, mhc_embedding, repr_layers=[], padding_mask=None, need_head_weights=False, return_contacts=False):
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        # if not padding_mask.any():
        #     padding_mask = None
        
        condition_embedding = self.condition_mlp(torch.cat([t, mhc_embedding], dim=-1))

        for layer_idx, layer in enumerate(self.conditional_layers):
            x, attn = layer(
                x,
                condition_embedding=condition_embedding,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.esm_model.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.esm_model.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}

        return result

        

    def forward(self, x, t, guidance_scale=1.0, mhc_embedding=None):
        # Embed time
        t = self.time_embedding(t)
        t = t.unsqueeze(1).repeat(1, self.length, 1)  # Repeat time embedding for each token

        # Embed tokens
        x = self.esm_model.embed_tokens(x)

        if self.training and torch.rand(1) < 0.1:  
            mhc_embedding = None


        conditional_bool = False
        if mhc_embedding is not None:
            mhc_embedding = self.mhc_mlp(mhc_embedding)
            mhc_embedding = mhc_embedding.unsqueeze(1).repeat(1, self.length, 1)

            conditional_bool = True

            if self.adaptive:
                h_cond = x
                # h_cond = h_cond.transpose(1, 2)
                result_cond = self.esm2_adaptive_forward(h_cond, t, mhc_embedding, repr_layers=[6])
                logits_cond = result_cond['logits']
            else:
                h_cond = x + t + mhc_embedding
                # h_cond = h_cond.transpose(1, 2)
                result_cond = self.esm2_forward(h_cond, repr_layers=[6])
                logits_cond = result_cond['logits']


        if self.adaptive:
            h_uncond = x
            # h_uncond = h_uncond.transpose(1, 2)
            result_uncond = self.esm2_adaptive_forward(h_uncond, t, torch.zeros_like(mhc_embedding), repr_layers=[6])
            logits_uncond = result_uncond['logits']
        else:
            h_uncond = x + t
            # h_uncond = h_uncond.transpose(1, 2)
            result_uncond = self.esm2_forward(h_uncond, repr_layers=[6])
            logits_uncond = result_uncond['logits']


        if conditional_bool:
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            logits = logits_uncond

        return logits