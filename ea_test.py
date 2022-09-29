# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import pandas as pd
from math import fabs,sqrt
import glob, os
from csv import writer
from scipy import stats

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# Run parameters
np.random.seed(123) # sets random seed for reproducibility
test_name = "boxplot_test"
experiment_lst = ["ea50_best", "ea25_best", "ea10_best"]
# experiment_lst = ["ea50_best"]

#####
########## Function definitions
#####

def individual_gain(p_e, e_e):
    """Compute individual gain"""
    return p_e - e_e

def derive_neurons(n_vars):
    """ Derive nr_hidden_neurons from n_vars in weights
    """
    return np.round((n_vars - 1 - 5) / (21 + 5)).astype("int")

def read_weights(path):
    """ Read best weights into dataframe
    """
    return pd.read_csv(path+".csv")

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def run_evoman(env, weights):
    """ Run evoman simulation for one individual weights list
        and return individual gain 
    """
    f,p,e,t = env.play(pcont=weights)
    return  individual_gain(p, e)

#####
########## Run Evoman simulation with best weights
#####

if __name__ == "__main__":

    # Prepare csv to save results
    df = pd.DataFrame({"Exp_name":[], "Enemy":[], "Run":[], "Gain":[]})
    if not os.path.exists(test_name+"csv"):
        df.to_csv(test_name+"_results.csv", index=False)
    
    # Loop over experiments
    for experiment in experiment_lst:
        df_weights = read_weights(experiment)

        # Loop over enemies
        for enemy in np.unique(df_weights["Enemy"]).astype("int"):
            # list of all 10 best weights for enemy
            test_weights = df_weights[df_weights["Enemy"] == enemy].loc[:, df_weights.columns != "Enemy"].to_numpy()
            # Loop over best weights for each run
            for run in range(test_weights.shape[0]):
                ig_list = []
                
                # Repeat evoman simulation 5 times and take mean
                for repeat in range(0, 5):
                    # Establish evoman environment
                    env = Environment(#experiment_name=experiment_name,
                                      enemies=[enemy],
                                      playermode="ai",
                                      player_controller=player_controller(derive_neurons(test_weights.shape[1])),
                                      enemymode="static",
                                      level=2,
                                      speed="fastest",
                                      logs="off")
                    
                    # Test weights against enemy and return individual gain
                    ig_list.append(run_evoman(env, np.asarray(test_weights[run])))
                
                mean_ig = np.mean(ig_list)

                # Print mean gain over 5 simulations in csv
                append_list_as_row(test_name+'_results.csv', [experiment, enemy, run, mean_ig])
    

    ### Run t-test on every enemy between ea50 and ea10
    exp1 = "ea50_best"
    exp2 = "ea10_best"

    df = pd.read_csv(test_name+'_results.csv')
    # df_neat = pd.read_csv()

    # Loop over each enemy
    for enemy in np.unique(df_weights["Enemy"]).astype("int"):
        ea50_result = np.asarray(df[(df["Exp_name"] == exp1) &
                                 (df['Enemy'] == enemy)].loc[:, 'Gain'].copy())
        ea10_result = np.asarray(df[(df["Exp_name"] == exp2) &
                                 (df['Enemy'] == enemy)].loc[:, 'Gain'].copy())
        # neat_result = np.asarray(df_neat[df['Enemy'] == enemy].loc[:, 'Gain'].copy())

        ttest = stats.ttest_ind(ea50_result, ea10_result, equal_var = False)
        
        # Print result of test to log
        print("T-test for difference in means between {} and {} for enemy:{} \n \
               test statistic = {}, p-value = {}".format(exp1, exp2, enemy, ttest[0], ttest[1]))
        
        """T-test for difference in means between ea50_best and ea10_best for enemy:3
                test statistic = -0.10939874362342053, p-value = 0.9140979641451267
           T-test for difference in means between ea50_best and ea10_best for enemy:6
                test statistic = 0.6148261752826397, p-value = 0.547349646422518
           T-test for difference in means between ea50_best and ea10_best for enemy:8
                test statistic = 0.4429802410305402, p-value = 0.6634697503546485
        """






                




    


