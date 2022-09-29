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

def select_mean_max(df, enemy):
    if df.name == 'Neat':
            df2 = df.loc[df['Enemy'] == enemy]
            mean_fit_gen = df2['Mean'].reset_index(drop=True)
            max_fit_gen = df2['Best'].reset_index(drop=True)
    else:
        df2 = df.loc[df['Enemy'] == enemy]
        mean_fit_gen = df2.groupby('Gen')['Mean_fit'].mean()
        max_fit_gen = df2.groupby('Gen')['Max_fit'].mean()
    return (mean_fit_gen, max_fit_gen)

def plot_enemy(total_df, enemy):
    """Creates a plot for each enemy, including all types of algorithms and their
       mean and max fitness values. 
    """
    # create the plot 
    plt.plot()
    plt.title(f"Performance of all algorithms for enemy {enemy}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    for df in total_df:
        # select the mean and max fitness for each generation
        mean_fit, max_fit = select_mean_max(df, enemy)
        # add mean and max values to the plot 
        add_mean_max(mean_fit, max_fit, df.name, df.color)
    
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
    
    neat = pd.read_csv('neat_results.csv')
    neat.name = "Neat"
    neat.color = "Yellow"

    total_df = [ea10, ea25, ea50, neat]

    enemies = [3, 6, 8]
    
    # create line plot with mean and max for each enemy
    for e in enemies:
        plot_enemy(total_df, e)

