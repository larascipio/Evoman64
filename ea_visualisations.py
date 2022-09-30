"""
ea_visualisations.py

Master AI: Evolutionary Computing

Uses the data coming from multiple experiments with two algorithms (NEAT and an EA), to create visualisations.
"""

# imports libraries 
import pandas as pd
from csv import reader, writer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# set global font for plots
plt.rcParams.update({'font.family':'sans-serif'})

def create_plot_enemy(total_df, enemy):
    """Creates a plot for each enemy, including all types of algorithms and their
       mean and max fitness values. 
    """
    # create the plot 
    plt.plot()
    plt.grid('on')
    plt.title(f'Performance of all algorithms for enemy {enemy}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    for df in total_df:
        # select the mean and max fitness for each generation
        mean_fit, max_fit = select_mean_max(df, enemy)
        # add mean and max values to the plot 
        add_mean_max(mean_fit, max_fit, df.name, df.color)
    
    custom_legend = [Line2D([0], [0], linestyle='--', color='black', lw=1),
                     Line2D([0], [0], linestyle='-',color='black', lw=1),
                     Line2D([0], [0], color=ea10.color, lw=1),
                     Line2D([0], [0], color=ea25.color, lw=1),
                     Line2D([0], [0], color=ea50.color, lw=1),
                     Line2D([0], [0], color=neat.color, lw=1)]

    plt.legend(custom_legend, ['Mean',
                               'Max', 
                               'EA-10', 
                               'EA-25', 
                               'EA-50', 
                               'Neat'])

    plt.savefig(f'line_plot_enemy_{enemy}.jpg')
    plt.cla()


def select_mean_max(df, enemy):
    """Selects all the mean and maximum fitness values for all 50 generations 
       for a specific enemy.
    """
    if df.name == 'Neat':
            df2 = df.loc[df['Enemy'] == enemy]
            mean_fit_gen = df2['Mean'].reset_index(drop=True)
            max_fit_gen = df2['Best'].reset_index(drop=True)
    else:
        df2 = df.loc[df['Enemy'] == enemy]
        mean_fit_gen = df2.groupby('Gen')['Mean_fit'].mean()
        max_fit_gen = df2.groupby('Gen')['Max_fit'].mean()
    return (mean_fit_gen, max_fit_gen)


def add_mean_max(mean_fit_gen, max_fit_gen, name, color):
    """Add a mean and max fitness value to a plot. 
    """
    plt.plot(mean_fit_gen, linestyle='--', color=color)
    plt.plot(max_fit_gen, linestyle='-', color=color, label=name)


def create_boxplot_enemy(df, enemy):
    """Creates three boxplots in a graph for a single enemy. 
    """
    # split data 
    df2 = df.loc[df['Enemy'] == enemy]
    best_ea_50 = df2.loc[df2['Exp_name'] == 'ea50_best']
    best_ea_25 = df2.loc[df2['Exp_name'] == 'ea25_best']
    best_ea_10 = df2.loc[df2['Exp_name'] == 'ea10_best']

    # create three boxplots in a graph
    labels = ['EA-10', 'EA-25', 'EA-50']
    colors = [ea10.color, ea25.color, ea50.color]

    plt.plot()
    plt.grid('on')
    plt.title(f'Distribution of individual gain for enemy {enemy}')
    plt.ylabel('Individual Gain')

    box = plt.boxplot([best_ea_10['Gain'], best_ea_25['Gain'], best_ea_50['Gain']], 
                 labels=labels, patch_artist=True)
    
    # color every boxplot differently
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.savefig(f'box_plot_enemy_{enemy}.jpg')
    plt.cla()


if __name__ == "__main__":
    # load csv files
    ea10 = pd.read_csv('ea10_results.csv')
    ea10.name = 'EA-10'
    ea10.color = '#FFAC33'

    ea25 = pd.read_csv('ea25_results.csv')
    ea25.name = 'EA-25'
    ea25.color = '#3380FF'

    ea50 = pd.read_csv('ea50_results.csv')
    ea50.name = 'EA-50'
    ea50.color = '#04AB5A'
    
    neat = pd.read_csv('neat_results.csv')
    neat.name = 'Neat'
    neat.color = '#A40C0C'

    ea_test_results = pd.read_csv('boxplot_test_results.csv')

    total_df = [ea10, ea25, ea50, neat]
    enemies = [3, 6, 8]

    # create line- and boxplots
    for e in enemies:
        create_plot_enemy(total_df, e)
        create_boxplot_enemy(ea_test_results, e)

