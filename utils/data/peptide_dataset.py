import torch
import pandas as pd
import json
import esm

    

class ComplexDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, max_len=15, vocab_size=24, 
                 mhc_embedding='data/mhc_embeddings_esm2_t6_8M_UR50D.pt',
                 mhc_index_file='data/allele_to_sequence.json', 
                ):
        self.data = pd.read_csv(csv_file)

        with open(mhc_index_file, 'r') as f:
            self.mhc_embedding_index = json.load(f)
        
        self.mhc_embedding = torch.load(mhc_embedding)

        self.mhc_name = self.data['allele'].tolist()
        self.mhc_seq = self.data['mhc_seq'].tolist()

        self.allele_to_index = {allele: idx for idx, allele in enumerate(self.mhc_embedding_index.keys())}
        
        self.sequences = self.data['peptide'].values
        self.max_len = max_len

        self.esm_dict = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 
                         'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13,
                         'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23}
        self.vocab_size = len(self.esm_dict)

        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # transfer the sequence to index, add [CLS] token (index 0) at the beginning and [EOS] token (index 2) at the end
        seq_idx = [0] + [self.esm_dict.get(aa, 0) for aa in sequence] + [2]
        
        # Padding
        if len(seq_idx) < self.max_len + 2:  # +2 for [CLS] and [EOS]
            # add [PAD] token (index 1)
            pad_len = self.max_len + 2 - len(seq_idx)
            seq_idx = seq_idx + [1] * pad_len
            padding_mask = [0] * (len(sequence) + 2) + [1] * pad_len
        else:
            seq_idx = seq_idx[:self.max_len + 2]
            padding_mask = [0] * (self.max_len + 2)

        seq_tensor = torch.tensor(seq_idx, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

        

        mhc_embedding = self.mhc_embedding[self.allele_to_index[self.mhc_name[idx]]]



        
        return seq_tensor, padding_mask, mhc_embedding
    

