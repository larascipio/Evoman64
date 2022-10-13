"""
generalist_visualisations.py

Evolutionary Computing: Task 2

Visualizes data coming from multiple experiments with an EA neural network. 
"""

# imports libraries 
import pandas as pd
from csv import reader, writer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sympy import E

# set global font for plots
plt.rcParams.update({'font.family':'sans-serif'})

def create_plot(list, enemies):
    """Creates a plot for each enemy, including all types of algorithms and their
       mean and max fitness values. 
    """
    # create the plot 
    plt.plot()
    plt.grid('on')
    plt.title(f'Performance of the EA with FPS or Elitism for enemies {enemies}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    for df in list:
        # select the mean and max fitness for each generation
        mean_fit, max_fit = select_mean_max(df)

        # add mean and max values to the plot 
        add_mean_max(mean_fit, max_fit, df.color)
    
    custom_legend = [Line2D([0], [0], linestyle='--', color='black', lw=1),
                     Line2D([0], [0], linestyle='-',color='black', lw=1),
                     Line2D([0], [0], color='#3380FF', lw=1),
                     Line2D([0], [0], color='#A40C0C', lw=1)
                     ]

    plt.legend(custom_legend, ['Mean',
                               'Max',
                               'FPS', 
                               'Elitism'])

    plt.savefig(f'gen_line_plot_{enemies}.jpg')
    plt.cla()


def select_mean_max(df):
    """Selects all the mean and maximum fitness values for all 50 generations 
       for a specific enemy.
    """

    mean_fit_gen = df.groupby('Gen')['Mean_fit'].mean()
    max_fit_gen = df.groupby('Gen')['Max_fit'].mean()
    return (mean_fit_gen, max_fit_gen)


def add_mean_max(mean_fit_gen, max_fit_gen, df_color):
    """Add a mean and max fitness value to a plot. 
    """
    plt.plot(mean_fit_gen, linestyle='--', color=df_color)
    plt.plot(max_fit_gen, linestyle='-', color=df_color)


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
    fps_368 = pd.read_csv('ea_fps_results_368.csv')
    fps_368.color = '#3380FF' #blue 

    fps_127 = pd.read_csv('ea_fps_results127.csv')
    fps_127.color = '#3380FF'
    
    elite_368 = pd.read_csv('ea_elite_results_368.csv')
    elite_368.color = '#A40C0C' #red

    elite_127 = pd.read_csv('ea_elite127_results.csv')
    elite_127.color = '#A40C0C'

    total_df = [[fps_368, elite_368], [fps_127, elite_127]]
    enemies = ['3, 6 and 8', '1, 2 and 7']

    # create line- and boxplots 
    for list, e in zip(total_df, enemies):
        create_plot(list, e)
        # create_boxplot(ea_test_results, e)

