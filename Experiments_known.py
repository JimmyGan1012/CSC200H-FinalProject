from Algorithms import *

def plot(history,title=""):
    ax = sns.heatmap(history["demo"].transpose())
    ax.set(title="Number Of Success Samples"+title)
    ax.set_ylabel("Demographic Group")
    ax.set_xlabel("Iteration")

    # plt.figure()
    # plt.title("Dataset Choice history(i)"+title)
    # plt.ylabel("Dataset Choice")
    # plt.xlabel("Iteration")
    # plt.plot(history["choice"], 'x')

    plt.figure()
    plt.title("k Choice history"+title)
    plt.ylabel("k Choice")
    plt.xlabel("Iteration")
    plt.plot(history["k"], 'x')



if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    print("random approach:")
    sim = Simulator()
    sim.Scenario_SimilarDataSet_Skewed_Distribution()
    history = random_approach(sim, k=50, display=False)
    print()
    #plot(history,title="[Random Approach]")

    print("known_distribution_baseline approach:")
    sim = Simulator()
    sim.Scenario_SimilarDataSet_Skewed_Distribution()
    history = known_distribution_baseline(sim,display=False)
    print()
    #plot(history,title="[Baseline Approach(k=1)]")

    print("Variable K Known Distribution[proposed Algorithm]: ")
    sim = Simulator()
    sim.Scenario_SimilarDataSet_Skewed_Distribution()
    history = known_distribution_variable_k(sim, maxk=50,display=False)
    print()
    plot(history, title="[Proposed Algorithm(variable k>1)]")

    plt.show()