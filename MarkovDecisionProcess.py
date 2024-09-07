import inspect
import numpy as np

class MarkovDecisionProcess:
    # Constructor
    def __init__(self, state, action, action_bounds_func, transition_func, reward_func, T):
        self._state = state
        self._action = action
        self._next_state = None
        self._reward = None
        self._t = 0

        self._state_trajectory = np.empty([len(state), T + 1])
        self._action_trajectory = np.empty([len(action), T + 1])
        self._reward_trajectory = np.empty([T + 1])

        self._action_bounds_func = action_bounds_func
        self._transition_func = transition_func
        self._reward_func = reward_func
        self._T = T

    # Getters
    def get_state(self):
        return self._state
    
    def get_action(self):
        return self._action

    def get_reward(self):
        return self._reward

    def get_next_state(self):
        return self._next_state
    
    def get_t(self):
        return self._t
    
    def get_T(self):
        return self._T
    
    # Setters
    def take_action(self, action):
        self._action = action
        self._reward = self._reward_func(self._state, self._action)
        self._next_state = self._transition_func(self._state, self._action)

    def take_step(self):
        self._state = self._next_state
        self._t += 1

    def save_step(self):
        self._state_trajectory[:,self._t] = self._state
        self._action_trajectory[:,self._t] = self._action
        self._reward_trajectory[self._t] = self._reward

    # Overwritting
    def __str__(self):
        return (f'Current state: {self._state} \n' +
                f'Current action: {self._action} \n' +
                f'Current reward: {self._reward} \n' +
                f'Transition function: {self._transition_func.__name__}' +
                    f'{inspect.signature(self._transition_func)} \n'
                f'Reward function: {self._reward_func.__name__}' +
                    f'{inspect.signature(self._reward_func)}')

