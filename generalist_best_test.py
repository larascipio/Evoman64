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
test_name = "boxplot_generalist_test"
experiment_lst = ["ea_fps_best_368", "ea_fps_best_127", "ea_elite_best_368", "ea_elite127_best"] # names of csv files to load weights from
experiment_names = ["FPS_368", "FPS_127", "Elite_368", "Elite_127"] # names to save in csv for test results
test_enemies = [1, 2, 3, 4, 5, 6, 7, 8] # enemy numbers to test against


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
    # return  individual_gain(p, e)
    return np.asarray(p), np.asarray(e)

#####
########## Run Evoman simulation with best weights
#####

if __name__ == "__main__":

    # Prepare csv to save results
    df = pd.DataFrame({"Exp_name":[], "Run":[], "Gain":[]})
    if not os.path.exists(test_name+"csv"):
        df.to_csv(test_name+"_results.csv", index=False)

    # record which set of weights had best performance (individual gain) over all enemies
    best_ig = float('-inf')


    
    # Loop over experiment weights from csv files
    for best_weight, experiment in zip(experiment_lst, experiment_names):
        df_weights = read_weights(best_weight)

        # Setup test environment
        env = Environment(#experiment_name=experiment_name,
                            enemies=test_enemies,
                            multiplemode='yes',
                            playermode="ai",
                            player_controller=player_controller(derive_neurons(df_weights.shape[1])),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            logs="on",
                            savelogs='no')
        
        # Loop over best weights for each run
        for run in range(df_weights.shape[0]):
            ig_list = []
                
            # Repeat evoman simulation 5 times and take mean
            for repeat in range(0, 5):      
                # Test weights against enemy and return individual gain
                # ig_list.append(run_evoman(env, np.asarray(df_weights.iloc[run,:])))
                p_lst, e_lst = run_evoman(env, np.asarray(df_weights.iloc[run,:]))
                ig_list.append(np.mean(individual_gain(p_lst, e_lst)))
                
            
            # Take mean ig over the 5 runs
            mean_ig = np.mean(ig_list)

            # check if mean_ig is best so far
            if best_ig <= mean_ig:
                best_ig = mean_ig
                best_weights = np.asarray(df_weights.iloc[run, :])
                best_individual = [experiment, run]

            # Print mean gain over 5 simulations in csv
            append_list_as_row(test_name+'_results.csv', [experiment, run, mean_ig])
    

    ### Run t-test on every enemy between FPS_368 and Elite_368
    fps_exps = ["FPS_368", "FPS_127"]
    elite_exps = ["Elite_368", "Elite_127"]

    df = pd.read_csv(test_name+'_results.csv')

    # Loop over exp group
    # for enemy in np.unique(df_weights["Enemy"]).astype("int"):
    #     ea50_result = np.asarray(df[(df["Exp_name"] == exp1) &
    #                              (df['Enemy'] == enemy)].loc[:, 'Gain'].copy())
    #     ea10_result = np.asarray(df[(df["Exp_name"] == exp2) &
    #                              (df['Enemy'] == enemy)].loc[:, 'Gain'].copy())
    
    for exp_1, exp_2 in zip(fps_exps, elite_exps):
        fps_result = np.asarray(df[df["Exp_name"] == exp_1].loc[:, 'Gain'].copy())
        elite_result = np.asarray(df[df["Exp_name"] == exp_2].loc[:, 'Gain'].copy())


        ttest = stats.ttest_ind(fps_result, elite_result, equal_var = False)
        
        # Print result of test to log
        print("T-test for difference in means between {} and {}\n \
               test statistic = {}, p-value = {}".format(exp_1, exp_2, ttest[0], ttest[1]))
        
        """
        T-test for difference in means between FPS_368 and Elite_368
                test statistic = -0.29721670637751896, p-value = 0.770183047080194
        T-test for difference in means between FPS_127 and Elite_127
                test statistic = -2.3260071495387167, p-value = 0.03190087323483168
        """


    ### Run the very best weights again against all enemies 5 times and report
    np.random.seed(123)

    # Print name and run of best individual's experiment
    print("Very best individual was Experiment: {}, Run: {}".format(best_individual[0], best_individual[1]))

    """
    Very best individual was Experiment: FPS_368, Run: 5
    """

    # Store best individuals weights in txt file
    np.savetxt("group64_best.txt", best_weights)

    # Store mean player energy and mean enemy energy for each enemy
    dct = {"Mean_pe":[], "Mean_ee":[]}

    # loop over each enemy
    for enemy in test_enemies:
        
        env = Environment(#experiment_name=experiment_name,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=player_controller(derive_neurons(best_weights.shape[0])),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        logs="off")
        
        pe_lst = [] # store player health per repetition
        ee_lst = [] # store enemy health per repetition

        # repeat test 5 times per enemy
        for rep in range(0, 5):
            f, p, e, t = env.play(best_weights)
            pe_lst.append(p)
            ee_lst.append(e)
        
        dct["Mean_pe"].append(np.mean(pe_lst))
        dct["Mean_ee"].append(np.mean(ee_lst))
    
    df_best = pd.DataFrame(dct)
    df_best.to_csv("best_individual_results.csv", index=False)






                




    


