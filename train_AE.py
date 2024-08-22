import torch
torch.manual_seed(42)
from utils import *
from model import AE
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import random_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Data(normalize=True)
train_len = int(0.8*len(data))
train_data, val_data = random_split(data, [train_len, len(data) - train_len])
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

ae = AE(88)
ae.to(device)
lossfn = MSELoss()
optimizer = Adam(ae.parameters(), lr=1e-4)
epochs = 60

def main():
    save_loss = 10
    for epoch in range(1, epochs+1):
        train_losses, val_losses = [], []
        loop_obj = tqdm(train_loader)
        ae.train()
        for data in loop_obj:
            x, _, y = data
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred_y = ae(x)

            loss = lossfn(pred_y, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            loop_obj.set_description_str(f"Epochs: {epoch}")
            loop_obj.set_postfix_str(f'Loss: {loss.item():0.3f}')

        print(f"Avg Train Loss: {sum(train_losses) / len(train_losses)}")

        ae.eval()
        for data in val_loader:
            with torch.no_grad():
                x, _, y = data
                x, y = x.to(device), y.to(device)
                
                pred_y = ae(x)
                loss = torch.nn.functional.cosine_similarity(pred_y, y, dim=1)
                val_losses.append(loss.item())

        print(f"Avg Val Loss: {sum(val_losses) / len(val_losses)}")

        if sum(val_losses) / len(val_losses) < save_loss:
            torch.save(ae.state_dict(), 'ae.pth')
            save_loss = sum(val_losses) / len(val_losses)

if __name__ == "__main__":
    main()