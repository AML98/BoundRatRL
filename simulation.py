import numpy as np
import matplotlib.pyplot as plt

def simulate(mdp, policy):

    T = mdp.get_T()

    for t in range(T):
        state = mdp.get_state().copy() # Copy to avoid object reference
        action = policy(state)
        
        mdp.set_action(action)
        mdp.set_reward()
        mdp.save_step()
        mdp.take_step()



