import torch
import torch.optim as optim

class ActorCritic:
    def __init__(self, mdp, value_network, policy_network,
                 alpha_value, alpha_policy, alpha_reward,
                 lambda_value=0, lambda_policy=0):
        self._mdp = mdp
        self._value_network = value_network
        self._policy_network = policy_network

        # Temporal difference 
        self._td_error = 0
        self._accum_error = 0
        self._alpha_reward = alpha_reward

        # Eligibility traces (lambda == 0 <--> no traces)
        if lambda_value:
            self._lambda_value = lambda_value
        if lambda_policy:
            self._lambda_policy = lambda_policy

        self._trace_value = {name: torch.zeros_like(param) for name, 
                             param in value_network.named_parameters()}
        self._trace_policy = {name: torch.zeros_like(param) for name, 
                              param in policy_network.named_parameters()}
        
        # Network optimizers
        self._alpha_value = alpha_value
        self._alpha_policy = alpha_policy

        self._optimizer_value = optim.Adam(value_network.parameters(), 
                                           lr = self._alpha_value)
        self._optimizer_policy = optim.Adam(policy_network.parameters(), 
                                            lr = self._alpha_policy)

    def update_td_error(self):
        reward = self._mdp.get_reward()
        state = torch.tensor(self._mdp.get_state(), dtype=torch.float32)
        next_state = torch.tensor(self._mdp.get_next_state(), dtype=torch.float32)
        
        state_value = self._value_network(state).item()
        next_state_value = self._value_network(next_state).item()
        self._td_error = reward - self._accum_error + state_value - next_state_value
        
    def update_accum_error(self):
        self._accum_error += self._alpha_reward * self._td_error
        
    def update_value_traces(self):
        for name, param in self._value_network.named_parameters():
            self._trace_value[name] = (self._lambda_value 
                * self._trace_value[name] + param.grad)
    
    def update_policy_traces(self):
        for name, param in self._policy_network.named_parameters(): 
            self._trace_policy[name] = (self._lambda_policy  
                * self._trace_policy[name] + param.grad)
            
    def update_value_network(self):
        for name, param in self._value_network.named_parameters():
            param.grad = self._trace_value[name] * self._td_error
        self._optimizer_value.step()

    def update_policy_network(self):
        for name, param in self._policy_network.named_parameters():
            param.grad = self._trace_policy[name] * self._td_error
        self._optimizer_policy.step()