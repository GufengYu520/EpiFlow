import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from sample_flow_matching import sample_flow_matching_discrete, decode_samples


def generate_sequences_for_all_alleles(allele_file='data/allele_100.txt', 
                                       output_file='data/result_sequences.csv',
                                       num_classes=33,
                                       esm_model='esm2_8m',
                                       input_esm_dim=320,
                                       n_samples_per_allele=100, 
                                       model_path=None, 
                                       guidance_scale=0.8, 
                                       step_size=0.1, 
                                       adaptive_guidance=False,
                                       mhc_embedding_path='data/mhc_embeddings_esm2_t6_8M_UR50D.pt',
                                       device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(allele_file, 'r') as f:
        alleles = [line.strip() for line in f if line.strip()]
    

    

    results = []
    
    for allele in tqdm(alleles):
        try:
            samples = sample_flow_matching_discrete(
                num_classes=num_classes,
                esm_model=esm_model,
                input_esm_dim=input_esm_dim,
                n_samples=n_samples_per_allele,
                step_size=step_size,
                device=device,
                model_path=model_path,
                conditional=True,
                mhc_allele=allele,
                guidance_scale=guidance_scale,
                adaptive_guidance=adaptive_guidance,
                mhc_embedding_path=mhc_embedding_path
            )
            
            decoded_seqs = decode_samples(samples)
            
            for seq in decoded_seqs:
                results.append({
                    'MHC_name': allele,
                    'peptide': seq
                })
                
        except Exception as e:
            print(f" {allele} {e}")
            continue
    
    df = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    

    df.to_csv(output_file, index=False)
    
    
    return df


def main():
    allele_file = 'data/allele_100.txt'
    output_file = 'results/output_test.csv'
    n_samples_per_allele = 100
    
    default_model_path = 'checkpoints/best_model_2.pt'

    df = generate_sequences_for_all_alleles(
        allele_file=allele_file,
        output_file=output_file,
        num_classes=33,
        esm_model='esm2_8m',
        input_esm_dim=320,
        n_samples_per_allele=n_samples_per_allele,
        model_path=default_model_path,
        guidance_scale=1,
        step_size=0.1,
    )
    


if __name__ == "__main__":
    main()