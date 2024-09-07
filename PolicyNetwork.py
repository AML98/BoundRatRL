import torch
import torch.nn as nn
import torch.distributions as distribution

class PolicyNetwork(nn.Module):
    def __init__(self, mdp, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        
        self._input_size = len(mdp.get_state())
        self._hidden_size = hidden_size
        self._output_size = len(mdp.get_action())
        
        self._fc = nn.Linear(self._input_size, self._hidden_size)
        self._mean_layer = nn.Linear(self._hidden_size, self._output_size)
        self._log_sd_layer = nn.Linear(self._hidden_size, self._output_size) # Use log to avoid negative SD

    def forward(self, x):
        x = self._fc(x)
        x = torch.relu(x)
        mean = self._mean_layer(x)
        log_sd = self._log_sd_layer(x)
        sd = torch.exp(log_sd)

        dist = distribution.Normal(mean, sd)

        return dist