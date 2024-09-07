import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, mdp, hidden_size=128):
        super(ValueNetwork, self).__init__()
        
        self._input_size = len(mdp.get_state())
        self._hidden_size = hidden_size
        self._output_size = 1 
        
        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._output_size)

    def forward(self, x):
        x = self._fc1(x)
        x = torch.relu(x)
        x = self._fc2(x)
        
        return x
