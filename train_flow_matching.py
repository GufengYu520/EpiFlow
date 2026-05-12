# Training flow matching
import os
import argparse
from datetime import datetime
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.tensorboard import SummaryWriter  
from flow_matching.path import MixtureDiscreteProbPath, AffineProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler, CondOTScheduler
from flow_matching.loss import MixturePathGeneralizedKL
import esm

from model.flow_matching_generator import ESM
from utils.data.peptide_dataset import ComplexDataset

torch.manual_seed(42)
np.random.seed(42)

def train_flow_matching_discrete(probability_denoiser, dataloader, device, args, epochs=100, num_classes=24, lr=1e-4, 
                        weight_decay=1e-5, epsilon=1e-3, patience=100, guidance_scale=0.5, save_step_interval=500):
    probability_denoiser.train()

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.model_type == 'esm':
        if args.adaptive:
            param_suffix = f'{args.flow_matching_type}_{args.model_type}_n{num_classes}_esm{args.input_esm_dim}_adaptive_guid{guidance_scale}_lr{lr:.0e}_batch{args.batch_size}_{current_time}'
        else:    
            param_suffix = f'{args.flow_matching_type}_{args.model_type}_n{num_classes}_esm{args.input_esm_dim}_guid{guidance_scale}_lr{lr:.0e}_batch{args.batch_size}_{current_time}'
    else:
        param_suffix = f'{args.flow_matching_type}_{args.model_type}_n{num_classes}_esm{args.input_esm_dim}_pro{args.pro_dim}_dim{args.hidden_dim}_time{args.time_dim}_guid{guidance_scale}_lr{lr:.0e}_batch{args.batch_size}_{current_time}'
    

    model_dir = os.path.join(args.model_dir, param_suffix)
    os.makedirs(model_dir, exist_ok=True)

    args_path = os.path.join(model_dir, 'training_args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    

    log_dir = os.path.join(args.log_dir, param_suffix)
    os.makedirs(log_dir, exist_ok=True)
    

    writer = SummaryWriter(log_dir=log_dir)

    # instantiate a convex path object
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)


    optimizer = torch.optim.Adam(probability_denoiser.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    loss_fn = MixturePathGeneralizedKL(path=path)


    best_loss = float('inf')
    counter = 0

    global_step = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, padding_mask, mhc) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.to(device)
            padding_mask = padding_mask.to(device)
            mhc = mhc.to(device)

            # init x_0 and x_1
            x_1 = data
            x_0 = torch.randint_like(x_1, high=num_classes)

            t = torch.rand(x_1.shape[0]).to(device) * (1 - epsilon)

            # sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

            # discrete flow matching generalized KL loss
            logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t, 
                                          guidance_scale=guidance_scale, mhc_embedding=mhc)

            loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(probability_denoiser.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            

            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)


            if global_step % save_step_interval == 0 and global_step > 0:
                step_save_path = os.path.join(model_dir, f'flow_matching_step{global_step}_loss{loss.item():.4f}.pth')
                torch.save(probability_denoiser.state_dict(), step_save_path)
                print(f"Step model saved at step {global_step} with loss: {loss.item():.4f} to {step_save_path}")
            
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        

        writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)
        

        lr_scheduler.step(avg_loss)
        

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(model_dir, f'flow_matching_best_epoch{epoch+1}_{avg_loss:.4f}.pth')
            torch.save(probability_denoiser.state_dict(), save_path)
            print(f"Best model saved with loss: {best_loss:.4f} to {save_path}")
            counter = 0
        else:
            counter += 1
            print(f"No improvement in loss for {counter} epochs")
            
            
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break



    writer.close()

    return best_loss



def parse_args():
    parser = argparse.ArgumentParser(description='Flow Matching Training for Peptide Generation')
    
    parser.add_argument('--time_dim', type=int, default=128, help='Time embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--input_esm_dim', type=int, default=320, help='Input ESM dimension')
    parser.add_argument('--pro_dim', type=int, default=128, help='Protein embedding dimension')
    parser.add_argument('--condition', type=bool, default=True, help='Condition on protein embedding')
    parser.add_argument('--esm_model', type=str, default='esm2_8m', choices=['esm2_8m', 'esm2_150m'], help='ESM model type')
    parser.add_argument('--num_classes', type=int, default=33, help='Number of classes')
    parser.add_argument('--adaptive', type=bool, default=False, help='Use adaptive ESM model')
    
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='Epsilon for time sampling')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale for conditional training')
    parser.add_argument('--save_step_interval', type=int, default=1000, help='Interval steps to save model checkpoint')
    
    parser.add_argument('--dataset_path', type=str, default='data/full_seq_dataset.csv', help='Path to training dataset')
    parser.add_argument('--mhc_embedding', type=str, default='data/mhc_embeddings_esm2_t6_8M_UR50D.pt', help='MHC embedding type')
    
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='Base directory for saving models')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Base directory for TensorBoard logs')
    
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda:id or cpu)')
    
    return parser.parse_args()



def main():
    args = parse_args()
    
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training parameters: {vars(args)}")
    

    train_dataset = ComplexDataset(args.dataset_path, mhc_embedding=args.mhc_embedding)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    
    probability_denoiser = ESM(
        esm_model=args.esm_model,
        input_esm_dim=args.input_esm_dim, 
        condition=args.condition,
        length=17,
        flow_matching_type=args.flow_matching_type,
        adaptive=args.adaptive
    ).to(device)


    
    print(f"Training {args.flow_matching_type} {args.model_type} flow matching model...")
    


    best_loss = train_flow_matching_discrete(
        probability_denoiser=probability_denoiser, 
        dataloader=train_loader, 
        device=device, 
        args=args,
        epochs=args.epochs, 
        num_classes=args.num_classes, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        epsilon=args.epsilon,
        patience=args.patience,
        guidance_scale=args.guidance_scale,
        save_step_interval=args.save_step_interval
    )
    
    
    print(f"Training completed. Best loss achieved: {best_loss:.4f}")
    
if __name__ == "__main__":
    main()