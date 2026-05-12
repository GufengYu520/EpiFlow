import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper

from model.sampler_RL import MixtureDiscreteEulerSolver
from model.BAPredictor import ESM_BApredictor
from utils.analysis_tools import calculate_instability  
from sample_flow_matching import decode_samples

class ConditionalWrappedModel_grpo(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        mhc_embedding_list = extras.get('mhc_embedding_list', None)
        guidance_scale = extras.get('guidance_scale', 0)
        
        if mhc_embedding_list is not None:
            mhc_embedding_all = []
            n_cond_per_step = len(mhc_embedding_list)
            for i in range(n_cond_per_step):
                mhc_embedding = mhc_embedding_list[i].repeat(len(x) // n_cond_per_step, 1)
                mhc_embedding_all.append(mhc_embedding)
            mhc_embedding_all = torch.cat(mhc_embedding_all, dim=0)

            return torch.softmax(self.model(x, t, guidance_scale=guidance_scale, mhc_embedding=mhc_embedding_all), dim=-1)
        else:
            return torch.softmax(self.model(x, t), dim=-1)



class GRPOTrainer:
    def __init__(self, raw_model, args):
        self.args = args
        self.device = args.device
        
        self.model = deepcopy(raw_model)
        
        self.ref_model = deepcopy(raw_model)
        self.ref_model.eval()
        
        if args.pretrained_path:
            self.load_pretrained_model(args.pretrained_path)
        
        self.update_ref_model()
        

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
            
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
      
        
        scheduler_polynomial = PolynomialConvexScheduler(n=1.0)
        self.path = MixtureDiscreteProbPath(scheduler=scheduler_polynomial)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model_type == 'esm':
            if args.rl_num == 2:
                self.experiment_name = f"GRPO_{args.model_type}_n{args.num_classes}_esm{args.input_esm_dim}_kl{args.kl_coef}_alpha{args.alpha}_step{args.step_size}_lr{args.learning_rate}_w{args.w_instability}_w{args.w_binding}_batch{args.n_cond_per_step}*{args.num_samples_per_cond}_{timestamp}"
            else:
                self.experiment_name = f"GRPO_{args.model_type}_n{args.num_classes}_esm{args.input_esm_dim}_kl{args.kl_coef}_alpha{args.alpha}_step{args.step_size}_lr{args.learning_rate}_batch{args.n_cond_per_step}*{args.num_samples_per_cond}_{timestamp}"
        else:
            self.experiment_name = f"GRPO_{args.model_type}_n{args.num_classes}_esm{args.input_esm_dim}_pro{args.pro_dim}_dim{args.hidden_dim}_time{args.time_dim}_kl{args.kl_coef}_guid{args.guidance_scale}_lr{args.learning_rate}_batch{args.n_cond_per_step}_{timestamp}"
        self.checkpoint_dir = os.path.join(args.save_dir, self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.log_dir = os.path.join(args.log_dir, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        with open(os.path.join(self.checkpoint_dir, 'grpo_config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        print(f"Experiment: {self.experiment_name}")
    
    def load_pretrained_model(self, model_path):
        print(f"Loading pretrained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

    
    def update_ref_model(self):
        self.ref_model.load_state_dict(self.model.state_dict())
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def load_mhc_embedding(self, allele_name):
        mhc_embeddings = torch.load(self.args.mhc_embedding_file)
        
        with open(self.args.allele_index_file, 'r') as f:
            allele_to_seq = json.load(f)
        
        allele_to_idx = {allele: idx for idx, allele in enumerate(allele_to_seq.keys())}
            
        mhc_idx = allele_to_idx[allele_name]
        mhc_embedding = mhc_embeddings[mhc_idx]
        
        return mhc_embedding
    


    def compute_reward(self, samples, mhc_allele=None, w_instability=0.5, w_binding=0.5, binding_type='esm', device=None):
        samples_seq = decode_samples(samples)
        normal_seq_index = [i for i, seq in enumerate(samples_seq) if 'X' not in seq and len(seq) >= 5]
        abnormal_seq_index = [i for i, seq in enumerate(samples_seq) if 'X' in seq or len(seq) < 5]
        samples_seq_normal = [samples_seq[i] for i in normal_seq_index]

        instability_scores = np.zeros(len(samples_seq))
        
        instability_scores_normal = calculate_instability(samples_seq_normal)
        instability_scores[normal_seq_index] = [score if score is not np.nan else 150.0 for score in instability_scores_normal]
        instability_scores[abnormal_seq_index] = np.array([150.0] * len(abnormal_seq_index))


        # binding affinity
        binding_affinities = np.zeros(len(samples_seq))

        binding_affinities_normal = ESM_BApredictor(samples_seq_normal, mhc_allele, device)
        binding_affinities[normal_seq_index] = binding_affinities_normal
        binding_affinities[abnormal_seq_index] = np.array([0] * len(abnormal_seq_index))


        rewards_instability = [0 if score <= 30.0 else 1 for score in instability_scores] 
        rewards_instability = -torch.tensor(rewards_instability, device=device)
        rewards_instability = (rewards_instability - rewards_instability.mean()) / (rewards_instability.std() + 1e-8)

        # 
        rewards_binding = torch.tensor(binding_affinities, device=device)      
        rewards_binding = (rewards_binding - rewards_binding.mean()) / (rewards_binding.std() + 1e-8)    
    
        rewards = w_instability * rewards_instability + w_binding * rewards_binding
        
        return rewards, -torch.tensor(instability_scores, device=device), torch.tensor(binding_affinities, device=device)
    
    def train_step(self, n_cond_per_step, mhc_allele_list=None, global_step=0):
        self.model.train()

        total_loss = 0
        batch_kl = 0
        batch_pg_loss = 0
        batch_reward = 0
        batch_reward_instability = 0
        batch_reward_binding = 0
        
        wrapped_model = ConditionalWrappedModel_grpo(self.model)
        solver = MixtureDiscreteEulerSolver(
            model=wrapped_model,
            path=self.path,
            vocabulary_size=self.args.num_classes
        )
        
        wrapped_ref_model = ConditionalWrappedModel_grpo(self.ref_model)
        solver.ref_model = wrapped_ref_model  
        
        extra_args = {}
        extra_args['mhc_embedding_list'] = []

        if mhc_allele_list is not None:
            for i in range(n_cond_per_step):
                mhc_allele = mhc_allele_list[i]
                mhc_embedding = self.load_mhc_embedding(mhc_allele)
                mhc_embedding = mhc_embedding.to(self.device)
                extra_args['mhc_embedding_list'].append(mhc_embedding)

        extra_args['mhc_embedding_list'] = torch.stack(extra_args['mhc_embedding_list'], dim=0)
        extra_args['guidance_scale'] = self.args.guidance_scale
        
        x_init = torch.randint(
            size=(n_cond_per_step * self.args.num_samples_per_cond, self.args.seq_length),
            high=self.args.num_classes,
            device=self.device
        )
        
        samples, log_probs, ref_log_probs, _, _, p_1t_record, p_1t_ref_record = solver.sample(
            x_init=x_init,
            step_size=self.args.step_size,
            train_batchsize=n_cond_per_step,
            # temperature=self.args.temperature,
            alpha=self.args.alpha,
            num_samples=self.args.num_samples_per_cond,
            verbose=False,
            time_grid=torch.tensor([0.0, 1 - self.args.epsilon], device=self.device),
            **extra_args
        )


        for i in range(n_cond_per_step):
            if self.args.rl_num == 2:
                rewards, instability_rewards, binding_rewards = self.compute_reward(samples[i * self.args.num_samples_per_cond : (i + 1) * self.args.num_samples_per_cond], 
                                          mhc_allele_list[i], 
                                          self.args.w_instability, 
                                          self.args.w_binding,
                                          self.args.binding_type,
                                          self.device)
                batch_reward_instability += instability_rewards.mean().item()
                batch_reward_binding += binding_rewards.mean().item()

            elif self.args.rl_num == 1:
                rewards = self.compute_reward(samples[i * self.args.num_samples_per_cond : (i + 1) * self.args.num_samples_per_cond], 
                                            mhc_allele_list[i], 
                                            self.args.w_instability, 
                                            self.args.w_binding,
                                            self.args.binding_type,
                                            self.device)
            
            batch_reward += rewards.mean().item()


            total_log_probs = log_probs[:, i, :, :].sum(dim=0).sum(dim=1) 
            total_ref_log_probs = ref_log_probs[:, i, :, :].sum(dim=0).sum(dim=1)
            

            p_1t_i = p_1t_record[:, i, :, :]
            p_1t_ref_i = p_1t_ref_record[:, i, :, :]
            p_1t_i = torch.clamp(p_1t_i, min=1e-10)
            p_1t_ref_i = torch.clamp(p_1t_ref_i, min=1e-10)

            log_p1_t_ref_i = torch.log(p_1t_ref_i)
            
            # Shape: [step_num, num_per_cond, seq_len]
            kl_per_token = F.kl_div(log_p1_t_ref_i, p_1t_i, reduction='none').sum(dim=-1)
            
            kl = kl_per_token.mean()
            batch_kl += kl.item()
            
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            

            # ratio = torch.exp(total_log_probs - total_ref_log_probs)
            # pg_loss1 = ratio * advantages
            # pg_loss2 = advantages * torch.clamp(ratio, min=1.0-self.args.clip_eps, max=1.0+self.args.clip_eps)
            # pg_loss = -torch.mean(torch.min(pg_loss1, pg_loss2))
            ratio = total_log_probs - total_ref_log_probs
            pg_loss = ratio * advantages
            pg_loss = -torch.mean(pg_loss)
            batch_pg_loss += pg_loss.item()
            
            total_loss += pg_loss + self.args.kl_coef * kl

        
        total_loss /= n_cond_per_step
        batch_kl /= n_cond_per_step
        batch_pg_loss /= n_cond_per_step
        batch_reward /= n_cond_per_step
        batch_reward_instability /= n_cond_per_step
        batch_reward_binding /= n_cond_per_step
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        
        self.writer.add_scalar('Training/Batch_Total_Loss', total_loss.item(), global_step)
        self.writer.add_scalar('Training/Batch_PG_Loss', batch_pg_loss, global_step)
        self.writer.add_scalar('Training/Batch_KL', batch_kl, global_step)
        self.writer.add_scalar('Training/Batch_Reward', batch_reward, global_step)
        if self.args.rl_num == 2:
            self.writer.add_scalar('Training/Batch_Reward_Instability', batch_reward_instability, global_step)
            self.writer.add_scalar('Training/Batch_Reward_Binding', batch_reward_binding, global_step)
        self.writer.add_scalar('Training/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)

        if global_step % self.args.save_step_interval == 0 and global_step > 0:
            if self.args.rl_num == 2:
                step_save_path = os.path.join(self.checkpoint_dir, f'flow_matching_step{global_step}_instability{batch_reward_instability:.4f}_binding{batch_reward_binding:.4f}.pth')
                print(f"Step model saved at step {global_step} with reward: instability {batch_reward_instability:.4f} binding {batch_reward_binding:.4f} to {step_save_path}")
            else:
                step_save_path = os.path.join(self.checkpoint_dir, f'flow_matching_step{global_step}_reward{batch_reward:.4f}.pth')
                print(f"Step model saved at step {global_step} with reward: {batch_reward:.4f} to {step_save_path}")
            torch.save(self.model.state_dict(), step_save_path)
        
        return {
            'total_loss': total_loss.item(),
            'pg_loss': batch_pg_loss,
            'kl': batch_kl,
            'rewards_mean': batch_reward,
            'rewards_instability_mean': batch_reward_instability,
            'rewards_binding_mean': batch_reward_binding,
        }


    def train(self, num_epochs, mhc_allele_list):
        best_reward = -float('inf')
        counter = 0
        global_step = 0

        mhc_allele_list_shuffled = mhc_allele_list.copy()
        
        
        for epoch in range(num_epochs):
            epoch_losses = 0
            epoch_rewards = 0
            epoch_rewards_instability = 0
            epoch_rewards_binding = 0
            epoch_pg_losses = 0
            epoch_kls = 0

            random.shuffle(mhc_allele_list_shuffled)
            
            steps_per_epoch = 100 // self.args.n_cond_per_step
            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx in range(steps_per_epoch):
                    metrics = self.train_step(
                        self.args.n_cond_per_step,
                        mhc_allele_list=mhc_allele_list_shuffled[batch_idx * self.args.n_cond_per_step : (batch_idx + 1) * self.args.n_cond_per_step],
                        global_step=global_step
                    )
                    
                    epoch_losses += metrics['total_loss']
                    epoch_rewards += metrics['rewards_mean']
                    epoch_rewards_instability += metrics['rewards_instability_mean']
                    epoch_rewards_binding += metrics['rewards_binding_mean']
                    epoch_pg_losses += metrics['pg_loss']
                    epoch_kls += metrics['kl']
                    
                    pbar.update(1)
                    global_step += 1
            
            avg_loss = epoch_losses / steps_per_epoch
            avg_reward = epoch_rewards / steps_per_epoch
            avg_reward_instability = epoch_rewards_instability / steps_per_epoch
            avg_reward_binding = epoch_rewards_binding / steps_per_epoch
            avg_pg_loss = epoch_pg_losses / steps_per_epoch
            avg_kl = epoch_kls / steps_per_epoch
            
            self.writer.add_scalar('Training/Epoch_Total_Loss', avg_loss, epoch)
            self.writer.add_scalar('Training/Epoch_PG_Loss', avg_pg_loss, epoch)
            self.writer.add_scalar('Training/Epoch_KL', avg_kl, epoch)
            self.writer.add_scalar('Training/Epoch_Reward', avg_reward, epoch)
            if self.args.rl_num == 2:
                self.writer.add_scalar('Training/Epoch_Reward_Instability', avg_reward_instability, epoch)
                self.writer.add_scalar('Training/Epoch_Reward_Binding', avg_reward_binding, epoch)

            
            self.scheduler.step(avg_loss)
            
            if self.args.rl_num == 2:
                print(f"Epoch {epoch+1} summary: Loss={avg_loss:.4f}, Reward_Instability={avg_reward_instability:.4f}, Reward_Binding={avg_reward_binding:.4f}")
            else:
                print(f"Epoch {epoch+1} summary: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")


            if avg_reward > best_reward:
                best_reward = avg_reward
                counter = 0
            else:
                counter += 1
                print(f"No improvement in loss for {counter} epochs")
                
                
            if counter >= self.args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.writer.close()
        
        print(f"Training completed. Best reward: {best_reward:.4f}")