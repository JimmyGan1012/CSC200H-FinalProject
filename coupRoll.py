import numpy as np

from Simulator import *


def known_k_couproll(sim: Simulator, ceil: int):
    history_demo = []
    history_choice = []

    while not sim.check_complete():
        x_min = float('inf')
        choice = None
        for j in range(sim.size_g):
            if sim.Current_Sampled_Count[j] >= sim.Desired_Counts[j]:
                continue
            j_source = np.zeros((sim.size_i, ceil))
            for i in range(sim.size_i):
                for k in range(ceil):
                    j_source[i][k] = min(k * sim.DS[i].DG_unused_count[j] / sim.DS[i].N,
                                 sim.Desired_Counts[j] - sim.Current_Sampled_Count[j]) / (sim.DS[i].c0 + sim.DS[i].cx * k)
            x_choice = np.unravel_index(np.argmax(j_source, axis=None), j_source.shape)
            x_max = j_source[x_choice[0]][x_choice[1]]
            if x_max < x_min:
                x_min = x_max
                choice = x_choice
            res = sim.sample(choice[0], k=choice[1]+1)
            if np.sum(res) == 0:
                res = -1
            history_demo.append(res)
            history_choice.append(choice)

    return np.array(history_demo, dtype=object), np.array(history_choice, dtype=object)


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    sim = Simulator()
    his_demo, his_choice = known_k_couproll(sim, 10)
    print(his_demo)
    print(his_choice)
