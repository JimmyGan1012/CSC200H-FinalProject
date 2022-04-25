from Simulator import *
import matplotlib.pyplot as plt
import seaborn as sns


def known_distribution_baseline(sim: Simulator):
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
            terms = [Dataset.DG_unused_count[j] / Dataset.N / Dataset.c0 for Dataset in sim.DS]
            max_term = np.max(terms)
            i = np.argmax(terms)
            if max_term < min:
                min = max_term
                choice = i
        res = sim.sample(choice, k=1)
        demographic_group_result = np.argmax(res)
        if np.sum(res) == 0:
            demographic_group_result = -1
        history["demo"].append(demographic_group_result)
        history["choice"].append(choice)
        history["cost"].append(sim.DS[choice].c0)
    for k in history.keys():
        history[k] = np.array(history[k])
    return history


def known_distribution_variable_k(sim: Simulator, maxk):
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
            for k in range(1, maxk + 1):
                terms = [k * Dataset.DG_unused_count[j] / Dataset.N / Dataset.get_cost(k) for Dataset in sim.DS]
                max_term = np.max(terms)
                i = np.argmax(terms)
                if max_term > max_for_k:
                    max_for_k = max_term
                    k_choice = k
                    i_choice = i
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


def random_approach(sim, k=1):
    history = dict()
    history["demo"] = []
    history["choice"] = []
    history["cost"] = []
    history["k"] = []

    while not sim.check_complete():
        choice = np.random.randint(0, len(sim.DS))
        res = sim.sample(choice, k=k)
        history["demo"].append(res)
        history["choice"].append(choice)
        history["cost"].append(sim.DS[choice].get_cost(k))
    for key in history.keys():
        history[key] = np.array(history[key])
    return history


def ucb_variable_k(sim: Simulator, maxk):
    history = dict()
    history["demo"] = []
    history["choice"] = []
    history["cost"] = []
    history["k"] = []

    R_plus_U = np.zeros((len(sim.DS),maxk),dtype=float)

    # First Round Query
    for i in range(len(sim.DS)):
        res = sim.sample(i, k=1)

    t = sim.size_i # Iteration counter
    while not sim.check_complete():
        # Calculate R+U for each i,k pair
        for i in range(len(sim.DS)):
            dataset = sim.DS[i]
            for k in range(1,maxk+1):
                #upperbond = k/ dataset.get_cost(k) * np.sqrt(2 * np.log(t) / (dataset.get_total_sampled_count()+1))
                reward = 0
                for g in range(1,k+1):
                    # Calculate duplicate prob
                    p_not_duplicate = ((dataset.N-dataset.get_total_sampled_count())/dataset.N)**g
                    # Calculate overflow prob
                    sum = 0.0
                    for j in range(sim.size_g):
                        sum += sim.Current_Sampled_Count[j]/sim.Desired_Counts[j]*sim.get_frequency(j)
                    p_not_overflow =(1-sum) ** g
                    #print("[IT {}] i={}, g={}, p_not_duplicate={}, p_not_overflow={}".format(t,i,g,p_not_duplicate,p_not_overflow))
                    assert p_not_duplicate > 0
                    assert p_not_overflow > 0
                    reward += g * p_not_duplicate * p_not_overflow
                    #print(reward)
                R_plus_U[i,k-1] = reward/ dataset.get_cost(k)
        max = R_plus_U.max()
        for i in range(len(sim.DS)):
            dataset = sim.DS[i]
            for k in range(1, maxk + 1):
                upperbond = max * np.sqrt(2 * np.log(t) / (dataset.get_total_sampled_count()+1))
                R_plus_U[i, k - 1] += upperbond


        unpacked = R_plus_U.argmax()
        i_choice =int( unpacked / R_plus_U.shape[1])
        k_choice = unpacked % R_plus_U.shape[1] +1
        res = sim.sample(i_choice, k=k_choice)

        print("Sampled {} samples from dataset {} with k={}".format(np.sum(res),i_choice,k_choice))
        history["demo"].append(res)
        history["choice"].append(i_choice)
        history["k"].append(k_choice)
        history["cost"].append(sim.DS[i_choice].get_cost(k_choice))
        t += 1
    for key in history.keys():
        history[key] = np.array(history[key])


    return history


if __name__ == "__main__":
    print("random approach:")
    history_rand = random_approach(Simulator(type=1), k=10)
    failure_rate = np.sum(history_rand['demo'] / (10 * len(history_rand["demo"])))

    print("Iteration Used:", len(history_rand["demo"]))
    print("Total Cost:", np.sum(history_rand["cost"]))
    print("Failure Rate(Sampled Useless Data):", failure_rate)

    # sim = Simulator(type=1)
    # history = known_distribution_baseline(sim)
    # failure_rate = np.count_nonzero(history["demo"] == -1) / len(history["demo"])
    #
    # print("Iteration Used:", len(history["demo"]))
    # print("Total Cost:", np.sum(history["cost"]))
    # print("Failure Rate(Sampled Useless Data):", failure_rate)
    #
    # i = 0
    # for DS in sim.DS:
    #     print(i, DS)
    #     i += 1
    #
    #
    # plt.figure()
    # plt.plot(history["demo"], 'x')
    #
    # plt.figure()
    # plt.plot(history["choice"], 'x')
    #
    # sim = Simulator(type=1)
    # history = known_distribution_variable_k(sim, 10)
    # failure_rate = np.sum(history['demo'] / (10 * len(history["demo"])))
    #
    # print("Iteration Used:", len(history["demo"]))
    # print("Total Cost:", np.sum(history["cost"]))
    # print("Failure Rate(Sampled Useless Data):", failure_rate)
    #
    # i = 0
    # for DS in sim.DS:
    #     print(i, DS)
    #     i += 1
    #
    # ax = sns.heatmap(history["demo"].transpose())
    # plt.figure()
    # plt.title("Dataset Choice history(i)")
    # plt.plot(history["choice"], 'x')
    #
    # plt.figure()
    # plt.title("k Choice history")
    # plt.plot(history["k"], 'x')
    #
    # plt.show()

    sim = Simulator(type=1)
    history = ucb_variable_k(sim,maxk=50)
    print("Total Successed Sample:",np.sum(history['demo']))
    print("Iteration Used:", len(history["demo"]))
    print("Total Cost:", np.sum(history["cost"]))
    failure_rate = (np.sum(history["k"]) - np.sum(history['demo']) )/ np.sum(history["k"])
    print("Failure Rate(Sampled Useless Data):", failure_rate)

    i = 0
    for DS in sim.DS:
        print(i, DS)
        i += 1

    ax = sns.heatmap(history["demo"].transpose())
    plt.figure()
    plt.title("Dataset Choice history(i)")
    plt.plot(history["choice"], 'x')

    plt.figure()
    plt.title("k Choice history")
    plt.plot(history["k"], 'x')
    plt.show()

