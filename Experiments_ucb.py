from Algorithms import *
import json
import os

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
    # Comment out next line to launch different tests
    seed = 1

    np.random.seed(seed)
    random.seed(seed)

    scenarios = ["Equal Distribution","Skewed Distribution","Very Skewed Distribution","Skewed Dataset Very Skewed Distribution"]
    apporachs = ["Random Approach","Baseline Approach(k=1)","Proposed Algorithm(variable k)"]

    if not os.path.exists("Result"):
        os.makedirs("Result")

    for scenario in [scenarios[-1]]:
        for apporach in [apporachs[-1]]:
            print(apporach," [",scenario,"]",sep="")
            sim = Simulator()
            if scenario == "Equal Distribution":
                sim.Scenario_SimilarDataSet_Equal_Distribution()
            elif scenario == "Skewed Distribution":
                sim.Scenario_SimilarDataSet_Skewed_Distribution()
            elif scenario == "Very Skewed Distribution":
                sim.Scenario_SimilarDataSet_Very_Skewed_Distribution()
            elif scenario == "Skewed Dataset Very Skewed Distribution":
                sim.Scenario_SkewedDataSet_Very_Skewed_Distribution()

            if apporach == "Random Approach":
                history = random_approach(sim, k=50, display=False)
            elif apporach == "Baseline Approach(k=1)":
                history = ucb_baseline(sim, display=False)
            elif apporach == "Proposed Algorithm(variable k)":
                history = ucb_variable_k(sim, maxk=50, display=True)

            history["seed"] = seed

            file_path = "Result/Unknown_Distribution_" + apporach.replace(" ","_")  + "_" + scenario.replace(" ","_") + ".json"
            for key in history.keys():
                if type(history[key]) == np.ndarray:
                    history[key] = history[key].tolist()
            with open(file_path,'w') as f:
                json.dump(history, f)
            print()