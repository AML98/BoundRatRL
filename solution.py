import numpy as np
import torch

print('check')

def solve(actor_critic):
    for t in range(actor_critic._mdp.get_T()):
        # Take action
        state = torch.tensor(actor_critic._mdp.get_state(), dtype=torch.float32)
        action = actor_critic._policy_network.sample_action(state)
        actor_critic._mdp.take_action(action)
        
        # Learn using Actor-Critic algorithm
        actor_critic.update_td_error()
        actor_critic.update_accum_error()

        if t != 0:
            actor_critic.update_value_traces()
            actor_critic.update_policy_traces()
        
        actor_critic.update_value_network()
        actor_critic.update_policy_network()

        # Go to next period and save
        actor_critic._mdp.take_step()
        actor_critic._mdp.save_step()

        if t % 10000 == 0 and t != 0:
            average_reward = np.mean(actor_critic._mdp._reward_trajectory[t-1000:t])
            print(f'Average reward at t = {t}: ', average_reward)
            print(f'State: ', actor_critic._mdp.get_state())