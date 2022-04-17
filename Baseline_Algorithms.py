from Simulator import *

def known_distribution_baseline(sim:Simulator):
    history = dict()
    history["demo"] = []
    history["choice"] = []
    history["cost"] = []

    while not sim.check_complete():
        min = float('inf')
        choice = None
        for j in range(sim.size_g):
            if sim.Current_Sampled_Count[j] >= sim.Desired_Counts[j]:
                continue
            terms = [Dataset.DG_unused_count[j]/Dataset.N / Dataset.c0 for Dataset in sim.DS]
            max_term = np.max(terms)
            i = np.argmax(terms)
            if max_term < min:
                min = max_term
                choice = i
        res = sim.sample(choice,k=1)
        demographic_group_result = np.argmax(res)
        if np.sum(res) == 0:
            demographic_group_result = -1
        history["demo"].append(demographic_group_result)
        history["choice"].append(choice)
        history["cost"].append(sim.DS[choice].c0)
    for k in history.keys():
        history[k] = np.array(history[k])
    return history

def known_distribution_variable_k(sim:Simulator,maxk):
    history = dict()
    history["demo"] = []
    history["choice"] = []
    history["cost"] = []
    history["k"] = []

    while not sim.check_complete():
        min = float('inf')
        final_i_choice = None
        final_k_choice = None
        for j in range(sim.size_g):
            if sim.Current_Sampled_Count[j] >= sim.Desired_Counts[j]:
                continue
            max_for_k = 0
            k_choice = None
            i_choice = None
            for k in range(1,maxk+1):
                terms = [k * Dataset.DG_unused_count[j] / Dataset.N / Dataset.get_cost(k) for Dataset in sim.DS]
                max_term = np.max(terms)
                i = np.argmax(terms)
                if max_term  > max_for_k:
                    max_for_k = max_term
                    k_choice=k
                    i_choice=i
            if max_for_k < min:
                min = max_for_k
                final_i_choice = i_choice
                final_k_choice = k_choice

        res = sim.sample(final_i_choice, k=final_k_choice)
        history["demo"].append(res)
        history["choice"].append(final_i_choice)
        history["k"].append(final_k_choice)
        history["cost"].append(sim.DS[final_i_choice].get_cost(final_k_choice))
    for key in history.keys():
        history[key] = np.array(history[key])
    return history



if __name__ == "__main__":
    # sim = Simulator(type=1)
    # history = known_distribution_baseline(sim)
    # failure_rate = np.count_nonzero(history["demo"] == -1) / len(history["demo"])
    #
    # print("Iteration Used:",len(history["demo"]))
    # print("Total Cost:", np.sum(history["cost"]))
    # print("Failure Rate(Sampled Useless Data):",failure_rate)
    #
    #
    # i=0
    # for DS in sim.DS:
    #     print(i,DS)
    #     i+=1
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(history["demo"],'x')
    #
    # plt.figure()
    # plt.plot(history["choice"],'x')
    # plt.show()

    sim = Simulator(type=1)
    history = known_distribution_variable_k(sim,10)
    failure_rate = np.sum(history['demo']/(10*len(history["demo"])))

    print("Iteration Used:",len(history["demo"]))
    print("Total Cost:", np.sum(history["cost"]))
    print("Failure Rate(Sampled Useless Data):", failure_rate)

    i=0
    for DS in sim.DS:
        print(i,DS)
        i+=1

    import matplotlib.pyplot as plt
    import seaborn as sns

    ax = sns.heatmap(history["demo"].transpose())


    plt.figure()
    plt.title("Dataset Choice history(i)")
    plt.plot(history["choice"],'x')

    plt.figure()
    plt.title("k Choice history")
    plt.plot(history["k"],'x')

    plt.show()
