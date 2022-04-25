from .Algorithms import *



def run_SimilarDataSet_Equal_Distribution():
    sim = Simulator()
    sim.Scenario_SimilarDataSet_Equal_Distribution()
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






if __name__ =="__main__":
    run_SimilarDataSet_Equal_Distribution()