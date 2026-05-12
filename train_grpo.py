import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import argparse
from tqdm import tqdm


from model.flow_matching_generator import ESM
from model.flow_matching_grpo import GRPOTrainer



def parse_args():
    parser = argparse.ArgumentParser(description='GRPO Training for Flow Matching')
    
    parser.add_argument('--model_type', type=str, default='esm', choices=['esm'], help='Model type')
    parser.add_argument('--num_classes', type=int, default=33, help='Number of amino acid classes')
    parser.add_argument('--seq_length', type=int, default=17, help='Peptide sequence length')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Model hidden dimension')
    parser.add_argument('--time_dim', type=int, default=8, help='Time embedding dimension')
    parser.add_argument('--input_esm_dim', type=int, default=320, help='Input ESM dimension')
    parser.add_argument('--pro_dim', type=int, default=128, help='Protein embedding dimension')
    parser.add_argument('--esm_model', type=str, default='esm2_8m', choices=['esm2_8m', 'esm2_150m'], help='ESM model type')
    parser.add_argument('--binding_type', type=str, default='esm', choices=['mhcflurry', 'esm'], help='Binding affinity type')
    
       
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--n_cond_per_step', type=int, default=4, help='Number of conditions per step')
    parser.add_argument('--num_samples_per_cond', type=int, default=32, help='Number of samples per condition')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='Epsilon for time sampling')
    # parser.add_argument('--num_time_steps', type=int, default=10, help='Number of time steps')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step size for flow matching solver')
    parser.add_argument('--kl_coef', type=float, default=0.01, help='KL divergence coefficient')
    # parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for categorical sampling')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha for Flow-GRPO schedule')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='Clip epsilon for policy gradient')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--save_step_interval', type=int, default=50, help='Interval to save step models')
    parser.add_argument('--patience', type=int, default=150, help='Patience for early stopping')
    parser.add_argument('--rl_num', type=int, default=2, help='Number of reward functions')
    parser.add_argument('--w_instability', type=float, default=1.0, help='Weight for instability reward')
    parser.add_argument('--w_binding', type=float, default=1.0, help='Weight for binding affinity reward')
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    parser.add_argument('--pretrained_path', type=str, default='./checkpoints/best_model_1.pt', help='Path to pretrained model')
    parser.add_argument('--mhc_embedding_file', type=str, default='data/mhc_embeddings_esm2_t6_8M_UR50D.pt', help='MHC embedding file path')
    parser.add_argument('--allele_index_file', type=str, default='data/allele_to_sequence.json', help='Allele index file path')
    

    parser.add_argument('--device', type=str, default='cuda:5', help='Device to use for training (cuda:id or cpu)')
    
    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # if args.model_type == 'esm':
    #     num_classes = 33
    # else:
    #     num_classes = 24

    print(f"Using device: {device}")
    print(f"Training parameters: {vars(args)}")

    
    probability_denoiser = ESM(
        esm_model=args.esm_model,
        input_esm_dim=args.input_esm_dim, 
        condition=True,
        length=17,
        ).to(device)
    

    trainer = GRPOTrainer(probability_denoiser, args)
    

    with open('./data/allele_100.txt', 'r') as f:
        mhc_allele_list = f.readlines()
        mhc_allele_list = [line.strip() for line in mhc_allele_list]


    trainer.train(
        num_epochs=args.num_epochs,
        mhc_allele_list=mhc_allele_list
    )

if __name__ == "__main__":
    main()