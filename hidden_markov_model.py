import numpy as np

# Distinct states of Markov process
states = np.array(["TV", "Pub", "Party", "Study"])
# Set of possible observations
observations = np.array(["Tired", "Hungover", "Scared", "Fine"])

# State Transition Matrix A has size (n_states, n_states)
#
# A = {a_ij} with i, j belong to {0, 1, ..., n_states-1}
#
# a_ij is the probability of state j given by previous state i,
# that is P(states[j] | states[i])
A = np.array([[0.4, 0.3, 0.1, 0.2],
              [0.6, 0.05, 0.1, 0.25],
              [0.7, 0.05, 0.05, 0.2],
              [0.3, 0.4, 0.25, 0.05]])

# Emission Matrix B has size (n_states, n_observations)
#
# B = {b_ij} with:
#   i belongs to {0, 1, ..., n_states-1}
#   j belongs to {0, 1, ..., n_observations-1}
#
# b_ij is the probability of observations j given by current state i,
# that is P(observations[j] | states[i])
B = np.array([[0.2, 0.1, 0.2, 0.5],
              [0.4, 0.2, 0.1, 0.3],
              [0.3, 0.4, 0.2, 0.1],
              [0.3, 0.05, 0.3, 0.35]])

# The sequence of observations {Tired, Tired, Scared}
x = [0, 0, 2]

# alpha[i, j] is the probability of the state_j at time t_i, given by x_0,.. x_t
alpha_matrix = np.ndarray((len(x), len(states)))
alpha_matrix[0, :] = [0.05, 0.1, 0.075, 0.075]


def compute_alpha(t, x_t, A, B, alpha_matrix):
    """
    Compute the joint probability of each state and observation x_t at time t.
    x_t is a index of 'observations' list.

    Arguments:
    t            -- Time t.
    x_t          -- The observation at time t.
    A            -- State Transition Matrix.
    B            -- Emission Matrix.
    alpha_matrix -- Alpha Matrix

    Return:
    alphas       -- A list of alpha values corresponding to each state y_t 
    """

    n_states = A.shape[0]
    alphas = np.zeros((1, n_states))

    for y_t in range(n_states):
        s = 0
        for previous_y_t in range(n_states):
            s += A[previous_y_t, y_t] * alpha_matrix[t - 1, previous_y_t]
        
        alphas[0, y_t] = B[y_t, x_t] * s

    # alphas is a matrix with size (1, n_states)
    return alphas


def compute_alpha_vec(t, x_t, A, B, alpha_matrix):
    """
    This is vectorized version of function 'compute_alpha'. It requires the same
    arguments and generates the same output also.

    Arguments:
    t            -- Time t.
    x_t          -- The observation at time t.
    A            -- State Transition Matrix.
    B            -- Emission Matrix.
    alpha_matrix -- Alpha Matrix

    Return:
    alphas       -- A list of alpha values corresponding to each state y_t 
    """

    # alphas is a matrix with size (1, n_states)
    alphas = B[:, x_t].T * np.dot(alpha_matrix[t - 1, :], A)

    return alphas

def find_the_most_likely_state_seq(alpha_matrix, state_labels=None):
    """
    Find the most likely series of states.
    It will re-use the result of Forward Algorithm (alpha_matrix).

    Arguments:
    alpha_matrix -- Alpha values computed by Forward Algorithm.
    state_labels -- List of state names.

    Return:
    state_seq    -- List of the most like states.
    """

    state_seq = np.argmax(alpha_matrix, axis=1)

    if state_labels is not None:
        state_seq = states[state_seq]

    return state_seq


if __name__ == "__main__":

    for t, x_t in enumerate(x):

        # We won't compute alpha for t = 0
        if t == 0:
            continue

        alphas = compute_alpha_vec(t, x_t, A, B, alpha_matrix).round(6)
        alpha_matrix[t, :] = alphas

    state_seq = find_the_most_likely_state_seq(alpha_matrix, states)

    print("Alpha matrix:")
    print(alpha_matrix.T)

    print("The most likely state sequence: " + str(state_seq))

