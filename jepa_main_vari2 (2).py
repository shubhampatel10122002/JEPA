import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import math
from typing import Tuple, Dict, NamedTuple, Optional

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor

class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float().to(self.device)
        actions = torch.from_numpy(self.actions[i]).float().to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)

        return WallSample(states=states, locations=locations, actions=actions)

def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        attention = self.conv(x)
        attention = torch.sigmoid(attention)
        return x * attention

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SpatialAttention(32),
            nn.MaxPool2d(2)  # 32x32
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SpatialAttention(64),
            nn.MaxPool2d(2)  # 16x16
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SpatialAttention(128),
            nn.MaxPool2d(2)  # 8x8
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SpatialAttention(256),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.projection = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(1)
        x = self.projection(x)
        return x

class Predictor(nn.Module):
    def __init__(self, embedding_dim=256, action_dim=2):
        super().__init__()
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, state_embed, action):
        action_embed = self.action_proj(action)
        combined = torch.cat([state_embed, action_embed], dim=-1)
        
        gate = self.gate(combined)
        transformed = self.transform(combined)
        
        output = gate * transformed + (1 - gate) * state_embed
        return output

class JEPAModel(nn.Module):
    def __init__(self, embedding_dim=256, momentum=0.99, temperature=0.1):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor(embedding_dim)
        self.repr_dim = embedding_dim
        self.momentum = momentum
        self.temperature = temperature
        
        # Initialize target encoder
        self.target_encoder = Encoder()
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # VicReg hyperparameters
        self.sim_coef = 25.0
        self.std_coef = 25.0
        self.cov_coef = 1.0

    @torch.no_grad()
    def _update_target_network(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, states, actions):
        """
        Args:
            states: Shape [B, T, 2, 64, 64] during training
                   Shape [B, 1, 2, 64, 64] during evaluation (only initial state)
            actions: Shape [B, T-1, 2]

        Returns:
            torch.Tensor: Shape [B, T, embedding_dim] containing state predictions
        """
        B = states.shape[0]
        T = actions.shape[1] + 1  # Total timesteps to predict
        predictions = []

        # Initial state encoding
        curr_state = self.encoder(states[:, 0])
        predictions.append(curr_state)

        # Always predict recurrently from current predicted state
        for t in range(T-1):
            curr_state = self.predictor(curr_state, actions[:, t])
            predictions.append(curr_state)

        # Return stacked predictions [B, T, embedding_dim]
        return torch.stack(predictions, dim=1)

    def compute_loss(self, states, actions):
        """Compute training loss using predictions against full state sequence"""
        B, T = states.shape[:2]
        predictions = self(states, actions)
        
        # Get target embeddings
        with torch.no_grad():
            targets = []
            for t in range(T):
                target = self.target_encoder(states[:, t])
                targets.append(target)
            targets = torch.stack(targets, dim=1)
        
        # Compute VicReg and InfoNCE losses for each timestep
        total_nce_loss = 0
        total_std_loss = 0
        total_cov_loss = 0
        
        for t in range(T):
            # InfoNCE loss
            pred_norm = F.normalize(predictions[:, t], dim=-1)
            target_norm = F.normalize(targets[:, t], dim=-1)
            sim = torch.matmul(pred_norm, target_norm.T) / self.temperature
            nce_loss = F.cross_entropy(sim, torch.arange(len(sim), device=sim.device))
            
            # VicReg variance loss
            std_loss = torch.sqrt(predictions[:, t].var(dim=0) + 1e-6)
            std_loss = torch.mean(F.relu(1 - std_loss))
            
            # VicReg covariance loss
            pred_centered = predictions[:, t] - predictions[:, t].mean(dim=0)
            cov = (pred_centered.T @ pred_centered) / (pred_centered.shape[0] - 1)
            cov_loss = (cov - torch.eye(cov.shape[0], device=cov.device)).pow(2).sum() / pred_centered.shape[-1]
            
            total_nce_loss += nce_loss
            total_std_loss += std_loss
            total_cov_loss += cov_loss
        
        # Combine losses with coefficients
        total_loss = (total_nce_loss + 
                     self.std_coef * total_std_loss + 
                     self.cov_coef * total_cov_loss) / T
        
        loss_components = {
            'nce_loss': total_nce_loss.item() / T,
            'std_loss': total_std_loss.item() / T,
            'cov_loss': total_cov_loss.item() / T,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
def train_jepa(
    data_path: str = "/scratch/DL24FA/train",
    batch_size: int = 256,
    num_epochs: int = 100,
    learning_rate: float = 2e-4,
    warmup_epochs: int = 5,
    device: str = "cuda",
    save_path: str = "model_weights.pth"
):
    # Create data loader
    train_loader = create_wall_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        device=device,
        train=True
    )
    
    model = JEPAModel(momentum=0.99).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return learning_rate * (epoch + 1) / warmup_epochs
        return learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        model.train()
        total_loss = 0
        total_nce_loss = 0
        total_std_loss = 0
        total_cov_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for idx, batch in enumerate(pbar, 1):
            optimizer.zero_grad()
            loss, loss_components = model.compute_loss(batch.states, batch.actions)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            
            optimizer.step()
            model._update_target_network()
            
            # Update statistics
            total_loss += loss_components['total_loss']
            total_nce_loss += loss_components['nce_loss']
            total_std_loss += loss_components['std_loss']
            total_cov_loss += loss_components['cov_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/idx:.4f}',
                'nce_loss': f'{total_nce_loss/idx:.4f}',
                'lr': f'{lr:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average NCE Loss: {total_nce_loss/num_batches:.4f}")
        print(f"Average Std Loss: {total_std_loss/num_batches:.4f}")
        print(f"Average Cov Loss: {total_cov_loss/num_batches:.4f}")
        print(f"Learning Rate: {lr:.6f}\n")
        
        # Save best model and early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model = train_jepa(
        batch_size=1024,  
        num_epochs=100,
        device=device
    )