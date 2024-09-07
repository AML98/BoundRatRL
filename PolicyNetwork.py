import torch
import torch.nn as nn
import torch.distributions as distribution

class PolicyNetwork(nn.Module):
    def __init__(self, mdp, hidden_size=128):
        super(PolicyNetwork, self).__init__()

        self._mdp = mdp
        
        self._input_size = len(mdp.get_state())
        self._hidden_size = hidden_size
        self._output_size = len(mdp.get_action())
        
        self._fc = nn.Linear(self._input_size, self._hidden_size)
        self._mean_layer = nn.Linear(self._hidden_size, self._output_size)
        self._log_sd_layer = nn.Linear(self._hidden_size, self._output_size) # Use log to avoid negative SD

        self._initialize_parameters()

    def _initialize_parameters(self):

        nn.init.constant_(self._mean_layer.weight, 0)
        nn.init.constant_(self._mean_layer.bias, 0)
        
        nn.init.constant_(self._log_sd_layer.weight, 0)
        nn.init.constant_(self._log_sd_layer.bias, 0)

    def forward(self, x):
        x = self._fc(x)
        x = torch.relu(x)
        mean = self._mean_layer(x)
        log_sd = self._log_sd_layer(x)
        log_sd = torch.clamp(log_sd, min=-10.0, max=2.0)
        sd = torch.exp(log_sd)

        dist = distribution.Normal(mean, sd)

        return dist
    
    def sample_action(self, state):
        dist = self.forward(state)
        unbounded_action = dist.sample()
        low, high = self._mdp._action_bounds_func(state)
        bounded_action = low + 0.5 * (high - low) * (torch.tanh(unbounded_action) + 1)

        return bounded_action