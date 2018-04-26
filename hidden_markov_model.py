import numpy as np


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


def compute_v(t, x_t, A, B, v_matrix):
    """
    A implementation of Viterbi algorithm.
    It returns the most probable sequence of states.

    Arguments:
    t               -- Time t.
    x_t             -- The observation at time t.
    A               -- State Transaction Matrix.
    B               -- Emission Matrix.
    v_matrix        -- Represents the Viterbi trellis

    Return:
    viterbi_output  -- A tuple that contains computed v_t and back_pointer.
    """

    n_states = A.shape[0]
    v_t = np.zeros((1, n_states))
    back_pointer = np.ndarray((1, n_states))

    for y_t in range(n_states):
        for previous_y_t in range(n_states):
            temp_v = A[previous_y_t, y_t] * B[y_t, x_t] * v_matrix[t-1, previous_y_t]
            if temp_v > v_t[0, y_t]:
                v_t[0, y_t] = temp_v
                back_pointer[0, y_t] = previous_y_t

    return (v_t, back_pointer)


if __name__ == "__main__":

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

    # Initial probabilities
    initial = [0.05, 0.1, 0.075, 0.075]

    # alpha_matrix[i, j] is the probability of the state_j at time t_i, given by x_0,.. x_t
    alpha_matrix = np.ndarray((len(x), len(states)))
    alpha_matrix[0, :] = initial

    # v_matrix[i, j] is the probability of the state_j at time t_i and passing
    # through the most probable state sequence of t_{0: t-1}.
    v_matrix = np.ndarray((len(x), len(states)))
    v_matrix[0, :] = initial

    back_pointer_matrix = np.ndarray(v_matrix.shape)
    back_pointer_matrix[0, :] = 0

    for t, x_t in enumerate(x):

        # We won't compute alpha for t = 0
        if t == 0:
            continue

        # Build Alpha matrix.
        alphas = compute_alpha_vec(t, x_t, A, B, alpha_matrix).round(6)
        alpha_matrix[t, :] = alphas

        # Build Viterbi matrix
        v_t, back_pointer = compute_v(t, x_t, A, B, v_matrix)
        v_matrix[t, :] = v_t
        back_pointer_matrix[t, :] = back_pointer

    # Backtrace process
    probable_seq = np.zeros(len(x), dtype=int)
    probable_seq[-1] = np.argmax(v_matrix[-1, :])
    for t in range(len(x) - 1, 0, -1):
        probable_seq[t - 1] = back_pointer_matrix[t, probable_seq[t]]

    probable_seq = states[probable_seq]

    print("Alpha matrix:")
    print(alpha_matrix.T)

    print("Viterbi trellis matrix:")
    print(v_matrix.T)

    print("Backpointer matrix:")
    print(back_pointer_matrix.T)

    print("The most probable state sequence: {}".format(str(probable_seq)))

