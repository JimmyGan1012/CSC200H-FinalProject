from Simulator import *

def known_distribution_baseline(sim:Simulator):
    history_demo = []
    history_choice = []

    while not sim.check_complete():
        for j in range(sim.size_g):
            min = float('inf')
            choice = None
            if sim.Current_Sampled_Count[j] >= sim.Desired_Counts[j]:
                continue
            max = 0
            max_index = None
            for i in range(sim.size_i):
                Dataset = sim.DS[i]
                term = Dataset.DG_unused_count[j]/Dataset.N / Dataset.c0
                if term > max:
                    max = term
                    max_index = i
            if max < min:
                min = max
                choice = max_index
            res = sim.sample(choice,k=1)
            demographic_group_result = np.argmax(res)
            if np.sum(res) == 0:
                demographic_group_result = -1
            history_demo.append(demographic_group_result)
            history_choice.append(choice)

    return np.array(history_demo) , np.array(history_choice)



if __name__ == "__main__":
    sim = Simulator(type=1)
    history_demo, history_choice = known_distribution_baseline(sim)
    failure_rate = np.count_nonzero(history_demo == -1) / len(history_demo)

    print("Iteration Used:",len(history_demo))
    print("Failure Rate(Sampled Useless Data):",failure_rate)


    i=0
    for DS in sim.DS:
        print(i,DS)
        i+=1

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history_demo,'x')

    plt.figure()
    plt.plot(history_choice,'x')
    plt.show()
