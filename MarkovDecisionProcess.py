class MarkovDecisionProcess:
    
    # Constructor
    def __init__(self, name, state, action, transition_func):
        self._name = name
        self._state = _State(state)
        self._action = _Action(action)
        self._transition = _Transition(transition_func)
        self._reward = None

    # Getters
    def get_name(self):
        return self._name
    
    def get_state(self):
        return self._state
    
    def get_action(self):
        return self._action
    
    def get_transition(self):
        return self._transition

    def get_reward(self):
        return self._reward
    
    # Setters
    def set_action(self, action):
        self._action = action

    def go_to_next_period(self):
        temp = self._transition._transition_func(self._state, self._action)
        self._reward = temp[1]
        self._state = temp[0]

class _State:
    def __init__(self, *state):
        self._state_vector = state

class _Action:
    def __init__(self, *action):
        self._action_vector = action

class _Transition:
    def __init__(self, transition_func):
        self._transition_func = transition_func

class _Policy:
    def __init__(self, policy_func):
        self._policy_func = policy_func

