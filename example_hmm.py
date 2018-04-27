import numpy as np
from hmm.hidden_markov_model import train, count_emission_pairs


if __name__ == "__main__":

    states = np.array(["F", "B"])
    observations = np.array(["H", "T"])

    data = np.array([[0, 0, 1, 0, 1, 0],
                     [1, 0, 1, 0, 1, 0],
                     [1, 0, 0, 1, 1, 0],
                     [1, 0, 1, 1, 1, 0],
                     [1, 0, 0, 1, 0, 1],
                     [0, 0, 1, 0, 0, 1],
                     [0, 0, 1, 1, 0, 1],
                     [0, 1, 1, 1, 0, 0]], dtype=int)


    state_sequences = np.array([[0, 1, 0, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0],
                                [1, 0, 0, 1, 1, 0],
                                [1, 0, 1, 1, 1, 0],
                                [1, 0, 0, 1, 0, 1],
                                [0, 0, 1, 0, 0, 1],
                                [0, 0, 1, 1, 0, 1],
                                [0, 1, 1, 1, 0, 0]], dtype=int)

    output = train(2, data, 1000)
    
    print(output["A"])
    print(output["B"])
    print(output["logs"][2])
    print(output["logs"][1][-1])

