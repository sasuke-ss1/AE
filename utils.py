import numpy as np
import torch
from torch.utils.data import Dataset

def parse_data(path='Data.txt'):
    X, Y = [], []

    with open(path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()  # Remove any leading/trailing whitespace
        
        if line:  # Check if the line is not empty
            
            # Split the line into tuple and the following number
            tuple_str, number_str = line.rsplit(') ', 1)
            tuple_str += ')'  # Add the closing parenthesis back to the tuple string
            
            # Convert the string representation of the tuple to a Python tuple
            values_tuple = list(eval(tuple_str))
            number = float(number_str)
            
            X.append(values_tuple)
            Y.append(number)

    return np.array(X), np.array(Y)

class Data(Dataset):
    def __init__(self, path='Data.txt', normalize=False):
        self.X, self.y = parse_data(path)
        self.normalize = normalize

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.normalize:
            return torch.tensor(self.X[idx] / np.sqrt(np.sum(self.X[idx]**2)), dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), torch.tensor(self.X[idx], dtype=torch.float32)
    
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), torch.tensor(self.X[idx], dtype=torch.float32)

