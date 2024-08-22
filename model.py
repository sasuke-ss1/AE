import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Scorer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, input_size, encode_size=16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, encode_size),
            nn.SiLU()
        )

    def forward(self, x):
        return self.model(x)
    
class Decoder(nn.Module):
    def __init__(self, output_size, encode_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(encode_size, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.model(x)


class AE(nn.Module):
    def __init__(self, input_size, encode_size=32, add_noise=False, mean=0, std=1):
        super().__init__()
        self.enc = Encoder(input_size, encode_size)
        self.dec = Decoder(input_size, encode_size)
        self.add_noise = add_noise
        self.mean = mean
        self.std = std

    def forward(self, x):
        emb = self.enc(x)
        noise = torch.normal(self.mean, self.std, emb.shape).to(device)

        if self.add_noise:
            emb = emb + noise

        return self.dec(emb)

