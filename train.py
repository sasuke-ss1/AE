import torch
torch.manual_seed(42)
from utils import *
from model import Scorer, AE
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss
from torch.utils.data import random_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Data(normalize=True)
train_len = int(0.8*len(data))
train_data, val_data = random_split(data, [train_len, len(data) - train_len])
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

scorer = Scorer(88)
scorer.load_state_dict(torch.load('scorer.pth'))
scorer.to(device)


ae = AE(88, add_noise=True)
ae.load_state_dict(torch.load('ae.pth'))
ae.to(device)

ae.enc.requires_grad_(False)
scorer.requires_grad_(False)

lossfn = L1Loss()
optimizer = Adam(scorer.parameters(), lr=1e-4)
epochs = 100

def main():
    save_loss = 10
    for epoch in range(1, epochs+1):
        train_losses, val_losses = [], []
        loop_obj = tqdm(train_loader)
        ae.train()
        for data in loop_obj:
            x, y, z = data
            x, y = x.to(device), y.to(device).unsqueeze(-1)
            
            optimizer.zero_grad()
            pred_x = ae(x)
            pred_reward = scorer(pred_x)
            loss = -pred_reward.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(-loss.item())
            loop_obj.set_description_str(f"Epochs: {epoch}")
            loop_obj.set_postfix_str(f'Reward: {-loss.item():0.3f}')

        print(f"Avg Reward: {sum(train_losses) / len(train_losses)}")

        ae.eval()
        for data in val_loader:
            with torch.no_grad():
                x, y, _ = data
                x, y = x.to(device), y.to(device).unsqueeze(-1)

                pred_x = ae(x)
                pred_reward = scorer(pred_x)
                loss = -pred_reward.mean()
                val_losses.append(-loss.item())

        print(f"Avg Reward: {sum(val_losses) / len(val_losses)}")

        if sum(val_losses) / len(val_losses) < save_loss:
            torch.save(scorer, 'ae_finetune.pth')
            save_loss = sum(val_losses) / len(val_losses)


if __name__ == "__main__":
    main()