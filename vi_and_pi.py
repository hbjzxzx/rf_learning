### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.
    
    r = R[state, :].reshape(action)
    action_2_next_state_prob = T[state, :, :]
    future = gamma * np.dot(action_2_next_state_prob, V) 
    
    backup_val = np.max(r + future, axis=0)
    
    return backup_val


def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    
    states_v = np.linspace(0, num_states-1, num=num_states, dtype=int)
    r = R[states_v, policy[states_v]]
    
    now_estimate_V = np.zeros(num_states)
    next_estimate_V = r + gamma * np.dot(T[states_v, policy[states_v], :], now_estimate_V) 

    while np.linalg.norm(next_estimate_V - now_estimate_V) > tol:
        now_estimate_V = next_estimate_V
        next_estimate_V = r + gamma * np.dot(T[states_v, policy[states_v], :], now_estimate_V) 
    
    value_function = next_estimate_V
    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    state_v = np.linspace(0, num_states-1, num=num_states, dtype=int)
    get_value_under_action = lambda a: gamma * np.dot(T[state_v, a, :], V_policy)
    
    future_expected = np.column_stack([get_value_under_action(a) for a in range(num_actions)])
    
    q_func = R + future_expected
    
    new_policy = np.argmax(q_func, axis=1)
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    policy = np.zeros(num_states, dtype=int)

    V_policy = policy_evaluation(policy, R, T, gamma, tol)
    new_policy = policy_improvement(policy, R, T, V_policy, gamma)
    while np.linalg.norm(new_policy - policy) > tol:
        policy = new_policy    

        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        new_policy = policy_improvement(policy, R, T, V_policy, gamma)
        
     
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    
    new_v = np.array(
        [bellman_backup(s, num_actions, R, T, gamma, value_function) for s in range(num_states)]
    )
    while np.linalg.norm(new_v - value_function) > tol:
        value_function = new_v
        new_v = np.array(
            [bellman_backup(s, num_actions, R, T, gamma, value_function) for s in range(num_states)]
        )

    value_function = new_v
        
    
    q = R + gamma * np.tensordot(T, value_function, axes=[[2],  [0]]).squeeze()
    policy = np.argmax(q, axis=1)
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'MEDIUM'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])
