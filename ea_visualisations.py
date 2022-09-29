"""
ea_visualisations.py

Master AI: Evolutionary Computing

Uses the data coming from multiple experiments with two algorithms (NEAT and an EA), to create visualisations.
"""

# imports libraries 
import pandas as pd
from csv import reader, writer
import matplotlib.pyplot as plt


def add_mean_max(mean_fit_gen, max_fit_gen, name, color):
    """Add a mean and max fitness value to a plot. 
    """
    plt.plot(mean_fit_gen, linestyle='--', color=color)
    plt.plot(max_fit_gen, linestyle='-', color=color, label=name)


def plot_enemy(total_df, enemy):
    """Creates a plot for each enemy, including all types of algorithms and their
       mean and max fitness values. 
    """
    plt.plot()
    plt.title(f"Performance of all algorithms for enemy {enemy}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    for df in total_df:
        df2 = df.loc[df['Enemy'] == enemy]
        mean_fit_gen = df2.groupby('Gen')['Mean_fit'].mean()
        max_fit_gen = df2.groupby('Gen')['Max_fit'].mean()
        add_mean_max(mean_fit_gen, max_fit_gen, df.name, df.color)
    plt.legend()
    plt.savefig(f'fig_enemy_{enemy}.jpg')
    plt.cla()


if __name__ == "__main__":
    # load csv files 
    ea10 = pd.read_csv('ea10_results.csv')
    ea10.name = "EA-10"
    ea10.color = "Blue"

    ea25 = pd.read_csv('ea25_results.csv')
    ea25.name = "EA-25"
    ea25.color = "Red"

    ea50 = pd.read_csv('ea50_results.csv')
    ea50.name = "EA-50"
    ea50.color = "Green"

    # df1 = pd.read_csv('ea10_best.csv')
    # df2 = pd.read_csv('ea10_best2.csv')
    # df_merged = df1.append(df2, ignore_index=True)
    # df_merged.to_csv('ea10_best.csv', index=False)
    # # df = pd.read_csv('ea10_results.csv')
    # # df.drop(columns=df.columns[0], axis=1, inplace=True)
    # # df.to_csv('ea10_results.csv', index=False)
    
    # neat = pd.read_csv('neat_results.csv')
    # neat.name = "Neat"
    # neat.color = "Blue"

    # total_df = [ea10, ea25, ea50, neat]
    total_df = [ea10, ea25, ea50]
    
    enemies = [3, 6, 8]
    
    # create line plot with mean and max for each enemy
    for e in enemies:
        plot_enemy(total_df, e)

