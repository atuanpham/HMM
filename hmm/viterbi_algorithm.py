import numpy as np


def compute_v(t, x_t, A, B, v_matrix):
    """
    Compute list of Viterbi Trellises of x_t.

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
            temp_v = A[previous_y_t, y_t] * B[y_t, x_t] * v_matrix[t - 1, previous_y_t]
            if temp_v > v_t[0, y_t]:
                v_t[0, y_t] = temp_v
                back_pointer[0, y_t] = previous_y_t

    return (v_t, back_pointer)


def find_most_likely_path(x, A, B, init_prob):
    """
    An implementation of Viterbi algorithm that finds the most probable
    sequence of states of a known observation sequence.

    Arguments:
    x               -- A sequence of observations.
    A               -- State Transition Matrix.
    B               -- Emission Matrix.
    init_prob       -- The initial probability of the Viterbi trellis matrix.

    Return:
    viterbi_output  -- A tuple that contains the computed Viterbi trellis
    matrix, backpointer matrix and the most likely state path.
    """

    # v_matrix[i, j] is the probability of the state_j at time t_i and passing
    # through the most probable state sequence of t_{0: t-1}.
    v_matrix = np.ndarray((len(x), A.shape[0]))
    v_matrix[0, :] = init_prob

    back_pointer_matrix = np.ndarray(v_matrix.shape)
    back_pointer_matrix[0, :] = 0

    for t, x_t in enumerate(x):

        # We don't compute alpha for t = 0
        if t == 0:
            continue

        # Build Viterbi matrix
        v_t, back_pointer = compute_v(t, x_t, A, B, v_matrix)
        v_matrix[t, :] = v_t
        back_pointer_matrix[t, :] = back_pointer

    # Backtrace process
    probable_seq = np.zeros(len(x), dtype=int)
    probable_seq[-1] = np.argmax(v_matrix[-1, :])
    for t in range(len(x) - 1, 0, -1):
        probable_seq[t - 1] = back_pointer_matrix[t, probable_seq[t]]

    return (v_matrix, back_pointer_matrix, probable_seq)

