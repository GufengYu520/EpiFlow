import torch
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from utils.data.peptide_dataset import ComplexDataset
import numpy as np
import json
import os
import esm

from model.flow_matching_generator import MLP, Unet1D, ESM

class ConditionalWrappedModel(ModelWrapper):
    # def __init__(self, model, n_samples=100):
    #     super().__init__(model)
    #     self.n_samples = n_samples

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        mhc_embedding = extras.get('mhc_embedding', None)
        guidance_scale = extras.get('guidance_scale', 0)
        
        if mhc_embedding is not None:
            
            mhc_embedding = mhc_embedding.repeat(len(x), 1)
                
            return torch.softmax(self.model(x, t, guidance_scale=guidance_scale, mhc_embedding=mhc_embedding), dim=-1)
        else:
            return torch.softmax(self.model(x, t), dim=-1)

def load_mhc_embedding(allele_name, embedding_file='data/mhc_embeddings_esm2_t6_8M_UR50D.pt', allele_index_file='data/allele_to_sequence.json'):
    mhc_embeddings = torch.load(embedding_file)
    
    with open(allele_index_file, 'r') as f:
        allele_to_seq = json.load(f)
    
    allele_to_idx = {allele: idx for idx, allele in enumerate(allele_to_seq.keys())}
        
    mhc_idx = allele_to_idx[allele_name]
    mhc_embedding = mhc_embeddings[mhc_idx]
    
    return mhc_embedding

def sample_flow_matching_discrete(num_classes=33, esm_model='esm2_8m', input_esm_dim=320, n_samples=100, step_size=0.1, device=None, 
                        model_path=None, conditional=False, 
                        mhc_allele=None, guidance_scale=1.0, adaptive_guidance=False,
                        mhc_embedding_path='data/mhc_embeddings_esm2_t6_8M_UR50D.pt'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dim = 17
    epsilon = 0.001
    

    
    # instantiate a convex path object
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    
    if num_classes == 33:
        probability_denoiser = ESM(esm_model, input_esm_dim=input_esm_dim, adaptive=adaptive_guidance).to(device)
    elif num_classes == 24:
        probability_denoiser = ESM(esm_model, input_esm_dim=input_esm_dim, adaptive=adaptive_guidance).to(device)
    probability_denoiser.load_state_dict(torch.load(model_path, map_location=device))
    probability_denoiser.eval()
    
    wrapped_probability_denoiser = ConditionalWrappedModel(probability_denoiser)
    solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=num_classes)
    
    x_init = torch.randint(size=(n_samples, dim), high=num_classes, device=device)
    
    extra_args = {}
    
    if conditional and mhc_allele is not None:
        mhc_embedding = load_mhc_embedding(mhc_allele, mhc_embedding_path)
        mhc_embedding = mhc_embedding.to(device)
        
        extra_args['mhc_embedding'] = mhc_embedding
        extra_args['guidance_scale'] = guidance_scale
        
        print(f"Conditional sampling: MHC allele: {mhc_allele}, guidance_scale: {guidance_scale}")
    else:
        print("Unconditional sampling")
    
    with torch.no_grad():
        sol = solver.sample(
            x_init=x_init, 
            step_size=step_size, 
            verbose=True, 
            # return_intermediates=True,
            time_grid=torch.linspace(0, 1 - epsilon, 10, device=device),
            **extra_args 
        )
    
    return sol



def decode_samples(samples):
    amino_acids = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                   'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

    
    decoded_seqs = []
    for sample in samples:
        seq = ''.join([amino_acids[token] if token < len(amino_acids) else 'X' for token in sample.cpu().numpy()])
        decoded_seqs.append(seq)
    
    decoded_seqs_str = [seq.replace('<cls>', '').replace('<eos>', '').replace('<pad>', '').replace('<unk>', '').replace('<mask>', '').replace('<null_1>', '') for seq in decoded_seqs]
    
    return decoded_seqs_str
