from Simulator import *
import matplotlib.pyplot as plt
import seaborn as sns


def known_distribution_baseline(sim: Simulator,display=False):
    history = dict()
    history["demo"] = []
    history["choice"] = []
    history["cost"] = []
    history["k"] = []

    while not sim.check_complete():
        min = float('inf')
        choice = None
        for j in range(sim.size_g):
            if sim.Current_Sampled_Count[j] >= sim.Desired_Counts[j]:
                continue
            terms = [dataset.DG_unused_count[j] / dataset.N / dataset.c0 for dataset in sim.DS]
            max_term = np.max(terms)
            i = np.argmax(terms)
            if max_term < min:
                min = max_term
                choice = i
        res = sim.sample(choice, k=1)
        if display:
            print("Sampled {} samples from dataset {} with k={}".format(np.sum(res), choice, 1))
        history["demo"].append(res)
        history["choice"].append(choice)
        history["k"].append(1)
        history["cost"].append(sim.DS[choice].get_cost(1))
    for key in history.keys():
        history[key] = np.array(history[key])
    print("Iteration Used:", len(history["demo"]))
    print("Total Cost:{:e}".format(np.sum(history["cost"])))
    failure_rate = (np.sum(history["k"]) - np.sum(history['demo'])) / np.sum(history["k"])
    print("Failure Rate(Sampled Useless Data):", failure_rate)
    return history

def known_distribution_variable_k(sim: Simulator, maxk,display=False):
    history = dict()
    history["demo"] = []
    history["choice"] = []
    history["cost"] = []
    history["k"] = []
    history["minority_choice"] = []

    while not sim.check_complete():
        min = float('inf')
        final_i_choice = None
        final_k_choice = None
        values_all_j = np.zeros((sim.size_g,len(sim.DS), maxk), dtype=float)
        for j in range(sim.size_g):
            if sim.Current_Sampled_Count[j] >= sim.Desired_Counts[j]:
                continue
            values = np.zeros((len(sim.DS), maxk), dtype=float)
            for i in range(sim.size_i):
                dataset = sim.DS[i]
                for k in range(1,maxk+1):
                    expected_number = k*(dataset.DG_total_count[j]- (dataset.DG_total_count[j]-dataset.DG_unused_count[j]) )/dataset.N
                    maximum_without_overflow = sim.Desired_Counts[j] - sim.Current_Sampled_Count[j]
                    #print("j={} ,i={}, k={}, expected_number:{}, maximum_without_overflow:{}".format(j,i,k,expected_number,maximum_without_overflow))
                    values[i][k-1] = np.min([expected_number,maximum_without_overflow])/dataset.get_cost(k)
            values_all_j[j] = values
            unpacked = values.argmax()
            i_choice = int(unpacked / values.shape[1])
            k_choice = unpacked % values.shape[1] + 1
            if values[i_choice][k_choice-1] < min:
                final_i_choice = i_choice
                final_k_choice = k_choice
                min = values[i_choice][k_choice-1]
                minority_target = j
        history["minority_choice"].append(minority_target)
        res = sim.sample(final_i_choice, k=final_k_choice)
        if display:
            print("Sampled {} samples from dataset {} with k={}".format(np.sum(res), final_i_choice, final_k_choice))
        history["demo"].append(res)
        history["choice"].append(final_i_choice)
        history["k"].append(final_k_choice)
        history["cost"].append(sim.DS[final_i_choice].get_cost(final_k_choice))
    for key in history.keys():
        history[key] = np.array(history[key])
    print("Iteration Used:", len(history["demo"]))
    print("Total Cost:{:e}".format(np.sum(history["cost"])))
    failure_rate = (np.sum(history["k"]) - np.sum(history['demo'])) / np.sum(history["k"])
    print("Failure Rate(Sampled Useless Data):", failure_rate)
    return history

def random_approach(sim, k=1,display=False):
    history = dict()
    history["demo"] = []
    history["choice"] = []
    history["cost"] = []
    history["k"] = []

    while not sim.check_complete():
        choice = np.random.randint(0, len(sim.DS))
        res = sim.sample(choice, k=k)
        if display:
            print("Sampled {} samples from dataset {} with k={}".format(np.sum(res), choice, k))
        history["demo"].append(res)
        history["choice"].append(choice)
        history["cost"].append(sim.DS[choice].get_cost(k))
        history["k"].append(k)
    for key in history.keys():
        history[key] = np.array(history[key])
    print("Iteration Used:", len(history["demo"]))
    print("Total Cost:{:e}".format(np.sum(history["cost"])))
    failure_rate = (np.sum(history["k"]) - np.sum(history['demo'])) / np.sum(history["k"])
    print("Failure Rate(Sampled Useless Data):", failure_rate)
    return history

def ucb_baseline(sim:Simulator,display=False):
    history = dict()
    history["demo"] = []
    history["choice"] = []
    history["cost"] = []
    history["k"] = []
    R_plus_U = np.zeros((len(sim.DS)), dtype=float)
    # First Round Query
    for i in range(len(sim.DS)):
        res = sim.sample(i, k=1)

    t = sim.size_i  # Iteration counter
    while not sim.check_complete():
        # Calculate R+U for each i,k pair
        for i in range(len(sim.DS)):
            dataset = sim.DS[i]
            reward = 0
            for j in range(sim.size_g):
                if sim.Current_Sampled_Count[j] >= sim.Desired_Counts[j]:
                    continue
                reward += (dataset.DG_total_count[j]-dataset.DG_unused_count[j])/sim.get_frequency(j)
            reward /= dataset.get_total_sampled_count() / dataset.get_cost(k=1)

            a = 0
            b = max([sim.get_frequency(j)/dataset.get_cost(k=1) for j in range(sim.size_g) if sim.Current_Sampled_Count[j] < sim.Desired_Counts[j]] )
            upperbound = (b-a) * np.sqrt(2*np.log(t)/dataset.get_total_sampled_count())
            R_plus_U[i] = reward + upperbound


        i_choice = R_plus_U.argmax()
        res = sim.sample(i_choice, k=1)
        if display:
            print("Sampled {} samples from dataset {} with k={}".format(np.sum(res), i_choice, 1))
        history["demo"].append(res)
        history["choice"].append(i_choice)
        history["k"].append(1)
        history["cost"].append(sim.DS[i_choice].get_cost(1))
        t += 1
    for key in history.keys():
        history[key] = np.array(history[key])
    print("Iteration Used:", len(history["demo"]))
    print("Total Cost:{:e}".format(np.sum(history["cost"])))
    failure_rate = (np.sum(history["k"]) - np.sum(history['demo'])) / np.sum(history["k"])
    print("Failure Rate(Sampled Useless Data):", failure_rate)
    return history

def ucb_variable_k(sim: Simulator, maxk,display=False):
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
                reward = 0
                for g in range(1,k+1):
                    # Calculate duplicate prob
                    p_not_duplicate = ((dataset.N-dataset.get_total_sampled_count())/dataset.N)**g
                    # Calculate overflow prob
                    sum = 0.0
                    for j in range(sim.size_g):
                        sum += sim.Current_Sampled_Count[j]/sim.Desired_Counts[j]*sim.get_frequency(j)
                    p_not_overflow =(1-sum) ** g
                    assert p_not_duplicate > 0
                    assert p_not_overflow > 0
                    reward += g * p_not_duplicate * p_not_overflow
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
        if display:
            print("Sampled {} samples from dataset {} with k={}".format(np.sum(res), i_choice, k_choice))
        history["demo"].append(res)
        history["choice"].append(i_choice)
        history["k"].append(k_choice)
        history["cost"].append(sim.DS[i_choice].get_cost(k_choice))
        t += 1
    for key in history.keys():
        history[key] = np.array(history[key])
    print("Iteration Used:", len(history["demo"]))
    print("Total Cost:{:e}".format(np.sum(history["cost"])))
    failure_rate = (np.sum(history["k"]) - np.sum(history['demo'])) / np.sum(history["k"])
    print("Failure Rate(Sampled Useless Data):", failure_rate)
    return history

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    print("random approach:")
    sim = Simulator()
    sim.Scenario_SkewedDataSet_Very_Skewed_Distribution()
    history_rand = random_approach(sim, k=50,display=False)
    print()

    # print("known_distribution_baseline approach:")
    # sim = Simulator()
    # sim.Scenario_SkewedDataSet_Very_Skewed_Distribution()
    # history_rand = known_distribution_baseline(sim,display=False)
    # print()
    #
    # print("Variable K Known Distribution")
    # sim = Simulator()
    # sim.Scenario_SkewedDataSet_Very_Skewed_Distribution()
    # history = known_distribution_variable_k(sim, maxk=50,display=False)
    # print()

    print("Unknown Baseline (UCB)")
    sim = Simulator()
    sim.Scenario_SkewedDataSet_Very_Skewed_Distribution()
    history = ucb_baseline(sim,display=True)

    print("Variable K Unknown Distribution(UCB)")
    sim = Simulator()
    sim.Scenario_SkewedDataSet_Very_Skewed_Distribution()
    history = ucb_variable_k(sim,maxk=50,display=True)

    # print(sim)

    # ax = sns.heatmap(history["demo"].transpose())
    # plt.figure()
    # plt.title("Dataset Choice history(i)")
    # plt.plot(history["choice"], 'x')
    #
    # plt.figure()
    # plt.title("k Choice history")
    # plt.plot(history["k"], 'x')
    #
    # plt.figure()
    # plt.title("minority_choice history")
    # plt.plot(history["minority_choice"], 'x')
    #
    # plt.show()

