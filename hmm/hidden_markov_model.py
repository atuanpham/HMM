import numpy as np
from .viterbi_algorithm import find_most_likely_path


def initialize_params(n_states, n_observations):

    A = np.random.rand(n_states, n_states)
    A = A / np.sum(A, axis=1, keepdims=True)
    B = np.random.rand(n_states, n_observations)
    B = B / np.sum(B, axis=1, keepdims=True)

    return (A, B)


def count_transition_pairs(state_sequences, start_, next_=None):

    count = 0
    for state_seq in state_sequences:
        # The last state of each sequence will be ignored
        for i in range(len(state_seq) - 1):

            if (state_seq[i] == start_
                    and (state_seq[i + 1] == next_ or next_ is None)):
                count += 1

    return count


def count_emission_pairs(state_sequences, obs, y, x=None):

    state_seq = np.array(state_sequences).flatten()
    ob_seq = np.array(obs).flatten()

    count = 0
    for i in range(state_seq.shape[0]):
        if state_seq[i] == y and (ob_seq[i] == x or x is None):
            count += 1

    return count


def update_params(n_states, n_observations, y, x):

    A = np.zeros((n_states, n_states))
    B = np.zeros((n_states, n_observations))

    for i in range(n_states):
        n_trans_start_with_i = count_transition_pairs(y, i)
        n_emis_start_with_i = count_emission_pairs(y, x, i)

        # Update Transition Matrix A
        for j in range(n_states):
            if n_trans_start_with_i != 0:
                A[i, j] = count_transition_pairs(y, i, j) / n_trans_start_with_i

        # Update Emission Matrix B
        for j in range(n_observations):
            if n_emis_start_with_i != 0:
                B[i, j] = count_emission_pairs(y, x, i, j) / n_emis_start_with_i 

    return (A, B)


def diff_of_params(param_1, param_2):

    A_1, B_1 = param_1
    A_2, B_2 = param_2
    
    flat_param_1 = np.concatenate([A_1.flatten(), B_1.flatten()])
    flat_param_2 = np.concatenate([A_2.flatten(), B_2.flatten()])

    diff = (1 / len(flat_param_1)) * np.sqrt(np.sum(np.power(flat_param_1 - flat_param_2, 2)))

    return diff


def train(n_states, obs, n_iter=50):
    
    n_observations = np.unique(obs).shape[0]

    # Initialization step:
    A, B = initialize_params(n_states, n_observations)
    initial_prob = np.ndarray(n_states)
    initial_prob[:] = 1 / n_states

    print("Init A:")
    print(A)
    print("Init B:")
    print(B)

    param_logs = []
    state_seq_logs = []
    diff_logs = []

    for i in range(n_iter):

        # Find the most likely state sequences corresponding to {A, B}
        state_sequences = []
        for o in obs:
            _, _, probable_seq = find_most_likely_path(o, A, B, initial_prob)
            state_sequences.append(probable_seq)

        A_next, B_next = update_params(n_states, n_observations, state_sequences, obs)
        diff = diff_of_params((A, B),
                              (A_next, B_next))
        A, B = A_next, B_next

        param_logs.append((A_next, B_next))
        state_seq_logs.append(state_sequences)
        diff_logs.append(diff)

        if diff == 0:
            break

    return {"A": A, "B": B,
            "logs": (param_logs, state_seq_logs, diff_logs)}

