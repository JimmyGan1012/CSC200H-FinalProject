import numpy as np
import pandas as pd
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

scenarios = ["Equal Distribution", "Skewed Distribution", "Very Skewed Distribution",
                 "Skewed Dataset Very Skewed Distribution"]
apporachs = ["Random Approach", "Baseline Approach(k=1)", "Proposed Algorithm(variable k)"]

def plot_known(df):
    sns.set(rc={'figure.figsize': (20, 8)})
    plt.figure()
    ax = sns.barplot(y="Total Cost", x="Scenario", hue="Methodology",
                     data=df[(df["Known or Unknown"] == "Known")])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    ax.set_title("Known Distribution Total Cost(10^6)")

    plt.figure()
    ax = sns.barplot(y="Total Queried Samples", x="Scenario", hue="Methodology",
                     data=df[(df["Known or Unknown"] == "Known")])
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Known Distribution Total Queried Samples")

    plt.figure()
    ax = sns.barplot(y="Duplicate Samples", x="Scenario", hue="Methodology",
                     data=df[(df["Known or Unknown"] == "Known")])
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Known Distribution Duplicate Samples")

    plt.figure()
    ax = sns.barplot(y="Overflow Samples", x="Scenario", hue="Methodology",
                     data=df[(df["Known or Unknown"] == "Known")])
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Known Distribution Overflow Samples")

def plot_unknown(df):
    sns.set(rc={'figure.figsize': (13, 6)})
    plt.figure()
    ax = sns.barplot(y="Total Cost", x="Scenario", hue="Methodology",
                     data=df[(df["Known or Unknown"] == "Unknown") & (df["Methodology"] != "Random Approach")])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    ax.set_title("Unknown Distribution Total Cost(10^6)")

    plt.figure()
    ax = sns.barplot(y="Total Queried Samples", x="Scenario", hue="Methodology",
                     data=df[(df["Known or Unknown"] == "Unknown") & (df["Methodology"] != "Random Approach")])
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Unknown Distribution Total Queried Samples")

    plt.figure()
    ax = sns.barplot(y="Duplicate Samples", x="Scenario", hue="Methodology",
                     data=df[(df["Known or Unknown"] == "Unknown") & (df["Methodology"] != "Random Approach")])
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Unknown Distribution Duplicate Samples")

    plt.figure()
    ax = sns.barplot(y="Overflow Samples", x="Scenario", hue="Methodology",
                     data=df[(df["Known or Unknown"] == "Unknown") & (df["Methodology"] != "Random Approach")])
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Unknown Distribution Overflow Samples")

def get_df():
    df = pd.DataFrame(columns=[
        "Total Cost",
        "Total Queried Samples",
        "Duplicate Samples",
        "Overflow Samples",
        "Known or Unknown",
        "Methodology",
        "Scenario"
    ])

    for scenario in scenarios:
        for apporach in apporachs:
            for Known_or_Unknown in ["Unknown","Known"]:
                filename = "Result/{}_Distribution_".format(Known_or_Unknown) + apporach.replace(" ", "_") + "_" + scenario.replace(" ",
                                                                                                              "_") + ".json"
                with open(filename) as json_file:
                    history = json.load(json_file)
                cost = np.sum(history["cost"])
                total_sample = np.sum(history["k"])
                dup_sample = history["duplicate_samples"]
                overflow_sample = history["overflow_samples"]

                df.loc[len(df)] = [cost / (10 ** 6), total_sample, dup_sample, overflow_sample, Known_or_Unknown, apporach,
                                   scenario]
    return df

def plot_ucb_k_history():
    apporach = apporachs[2]
    for scenario in scenarios:
        filename = "Result/Unknown_Distribution_" + apporach.replace(" ","_") + "_" + scenario.replace(" ","_") + ".json"
        with open(filename) as json_file:
            history = json.load(json_file)
        plt.figure()
        plt.title(scenario +"[Proposed Algorithm]")
        plt.plot(history["k"], 'o', label='variable k')
        plt.legend()

def plot_known_k_history_and_gets():
    apporach = apporachs[2]
    scenario = scenarios[2]
    filename = "Result/Known_Distribution_" + apporach.replace(" ","_") + "_" + scenario.replace(" ","_") + ".json"
    with open(filename) as json_file:
        history = json.load(json_file)

    plt.figure(figsize=[9,6])
    plt.title("K Choice History["+scenario+"]")
    plt.plot(history["k"], 'o', label='variable k')
    plt.ylabel("k")
    plt.xlabel("Iterations")

    scenario = scenarios[1]
    filename = "Result/Known_Distribution_" + apporach.replace(" ","_") + "_" + scenario.replace(" ","_") + ".json"
    with open(filename) as json_file:
        history = json.load(json_file)
    plt.figure(figsize=[9,6])
    ax = sns.heatmap(np.array(history["demo"]).transpose())
    ax.set(title="Successfully Queried Sample History[Proposed Algorithm," + scenario + "]")
    ax.set_ylabel("Demographic Group")
    ax.set_xlabel("Iteration")

    apporach = apporachs[0]
    filename = "Result/Known_Distribution_" + apporach.replace(" ","_") + "_" + scenario.replace(" ","_") + ".json"
    with open(filename) as json_file:
        history = json.load(json_file)
    plt.figure(figsize=[9,6])
    ax = sns.heatmap(np.array(history["demo"]).transpose())
    ax.set(title="Successfully Queried Sample History[Random Approach," + scenario + "]")
    ax.set_ylabel("Demographic Group")
    ax.set_xlabel("Iteration")


if __name__ =="__main__":
    sns.set_theme()
    df = get_df()
    print(df)

    plot_known(df)
    plot_unknown(df)

    plot_ucb_k_history()
    plot_known_k_history_and_gets()
    plt.show()

