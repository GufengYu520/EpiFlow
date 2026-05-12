import torch
import torch.nn as nn
import esm
import numpy as np
from mhcflurry import Class1PresentationPredictor
import json

class BAPredictor(nn.Module):
    def __init__(self, 
                esm_type='esm2_t6_8M_UR50D',  
                esm_hidden_dim=320,  
                esm_layer=6,  
                mlp_hidden_dim=128, 
                mlp_out_dim=1, 
                mlp_dropout=0.1,
            ):

        super(BAPredictor, self).__init__()

        self.esm_type = esm_type
        self.esm_hidden_dim = esm_hidden_dim
        self.esm_layer = esm_layer

        if esm_type == 'esm2_t6_8M_UR50D':
            self.mhc_esm, _ = esm.pretrained.esm2_t6_8M_UR50D()
            self.peptide_esm, _ = esm.pretrained.esm2_t6_8M_UR50D()
        else:
            raise ValueError(f"Unsupported ESM type: {esm_type}")
         
        self.esm_projection = nn.Linear(esm_hidden_dim + esm_hidden_dim, mlp_hidden_dim)

        self.seq_fc = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim // 2, mlp_out_dim)
        )
    
    def encode_with_esm(self, tokens, tokens_lens, chain_type):
        if chain_type == 'mhc':
            outputs = self.mhc_esm(tokens, repr_layers=[self.esm_layer], return_contacts=False)
        elif chain_type == 'peptide':
            outputs = self.peptide_esm(tokens, repr_layers=[self.esm_layer], return_contacts=False)
        else:
            raise ValueError(f"Unknown chain_type: {chain_type}")

        reprs = outputs["representations"][self.esm_layer] # (batch, batch_max_len, esm_hidden_dim)

        seq_reprs = []
        for i, length in enumerate(tokens_lens):
            seq_reprs.append(reprs[i, 1:length-1].mean(dim=0))

        seq_emb = torch.stack(seq_reprs, dim=0)  # (batch, esm_hidden_dim)

        return seq_emb

    def forward(self, mhc_tokens, mhc_lens, peptide_tokens, peptide_lens):
        '''
        mhc_tokens: (batch_size, mhc_seq_len)
        peptide_tokens: (batch_size, peptide_seq_len)
        '''
        mhc_rep = self.encode_with_esm(mhc_tokens, mhc_lens, chain_type='mhc')  # (batch_size, esm_emb_dim)
        peptide_rep = self.encode_with_esm(peptide_tokens, peptide_lens, chain_type='peptide')  # (batch_size, esm_emb_dim)
                
        seq_emb = torch.cat([mhc_rep, peptide_rep], dim=-1)  # (batch_size, esm_emb_dim + esm_emb_dim)

        seq_emb_proj = self.esm_projection(seq_emb)  # (batch_size, mlp_hidden_dim)

        out = self.seq_fc(seq_emb_proj)  # (batch_size, mlp_out_dim=1)

        out_norm = torch.sigmoid(out).squeeze(-1)  # (batch,) a value between 0 and 1

        return out_norm
    



def ESM_BApredictor(peptides, mhc_name, device):
    predictor = BAPredictor(
        esm_type='esm2_t6_8M_UR50D',
        esm_hidden_dim=320,
        esm_layer=6,
        mlp_hidden_dim=128,
        mlp_out_dim=1,
        mlp_dropout=0.1
    ).to(device)

    checkpoint = torch.load('checkpoints/BA_predictor_esm2_t6_8M_UR50D.pt', map_location=device)
    predictor.load_state_dict(checkpoint['model_state_dict'])
    predictor.eval()

    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    with open('data/allele_to_sequence.json', 'r') as f:
        allele_to_sequence = json.load(f)
    mhc_seq = allele_to_sequence[mhc_name]

    batch_mhc = [(f'sample_{i}', mhc_seq) for i in range(len(peptides))]
    batch_peptide = [(f'sample_{i}', peptide) for i, peptide in enumerate(peptides)]

    _, _, batch_mhc_tokens = batch_converter(batch_mhc)
    batch_mhc_lens = (batch_mhc_tokens != alphabet.padding_idx).sum(1)
    _, _, batch_peptide_tokens = batch_converter(batch_peptide)
    batch_peptide_lens = (batch_peptide_tokens != alphabet.padding_idx).sum(1)

    batch_mhc_tokens = batch_mhc_tokens.to(device)
    batch_mhc_lens = batch_mhc_lens.to(device)
    batch_peptide_tokens = batch_peptide_tokens.to(device)
    batch_peptide_lens = batch_peptide_lens.to(device)

    with torch.no_grad():
        
        affinity = predictor(batch_mhc_tokens, batch_mhc_lens, batch_peptide_tokens, batch_peptide_lens)
        affinity = affinity.cpu().numpy()

    
    return affinity


