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

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


#####
########## Parameter settings
#####

# 50 neurons, 10 runs, 100 individuals = 516 min runtime

# Change for each different experiment
n_hidden_neurons = 10 # neurons in MLP hidden layer
experiment_name = 'ea_elite'
n_runs = 10 # nr of complete algorithm cycles
enemy_lst = [3, 6, 8] # enemy to fight
top_nr = 5 # set number of best "elite" parents to save for next generation

# domain of weights
dom_u = 1
dom_l = -1

# domain of initial mutation step sizes
step_u = 3
step_l = 1

npop = 50  # population size
ngens = 30   # nr of generations per run
alpha = 0.5 # blend crossover parameter
# mutation = 0.2 # mutation probability not currently used in uncorrelated mutation
last_best = 0 # saves best fitness
np.random.seed(123) # sets random seed for reproducibility


#####
########## Environment settings & controller
#####

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(#experiment_name=experiment_name,
                  enemies=enemy_lst,
                  multiplemode='yes',
                  savelogs='no',
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

ini = time.time()  # sets time marker

#####
########## Evolution Functions
#####

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# normalization for selection probabilities
def norm(x, pfit_pop):
    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

# evaluation
def evaluate(x):
    # return np.array(list(map(lambda y: simulation(env,y), x[:, :n_vars])))
    return np.array(list(map(lambda y: simulation(env,y), x)))

# tournament selection
def tournament(pop):
    c1 =  np.random.randint(0,pop.shape[0], 1)
    c2 =  np.random.randint(0,pop.shape[0], 1)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]

# limits
def limits(x):
    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x

# uncorrelated mutation operator
def uncor_muta(offspring):
    """Adaptive uncorrelated mutation for n_var stepsizes
    """
    new_stepsize = np.zeros((n_vars)) # store step size for each weight

    #random base mutation for all step sizes
    base_step_muta = np.random.normal(0, (1/np.sqrt(2*n_vars)), 1)

    for gene in range(n_vars):
        #gene specific mutation
        gene_step_muta = np.random.normal(0, (1/np.sqrt(2*np.sqrt(n_vars))), 1)

        #mutate old stepsize with base and gene mutation
        new_stepsize[gene] = offspring[n_vars + gene] * np.exp(base_step_muta + gene_step_muta)

        #boundary rule to prevent stepsize close to 0
        new_stepsize = np.array([0.01 if i < 0.01 else i for i in new_stepsize])

        #mutate individual gene using new stepsize
        offspring[gene] = offspring[gene] + np.random.normal(0, new_stepsize[gene])
    
    #update the new stepsizes in the offspring
    offspring[n_vars:] = new_stepsize

    return offspring

# blend crossover
def blx(pop):

    total_offspring = np.zeros((0, pop.shape[1]))
    eps = (1-2*alpha)*np.random.uniform(0, 1) - alpha
    
    for p in range(0, pop.shape[0], 2):
        p1 = tournament(pop)
        p2 = tournament(pop)

        n_offspring = 1
        offspring = np.zeros((n_offspring, pop.shape[1]))

        for f in range(0, n_offspring):
            #blend crossover with alpha=0.5
            offspring[f] = (1-eps)*p1 + eps*p2

            #adaptive mutation
            offspring[f] = uncor_muta(offspring[f])
            
            #enforce upper and lower bound of weights
            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring


#####
########## Function to store results in csv file
#####

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


#####
########## Experiment simulation
#####
if __name__ == "__main__":
    for run in range(n_runs):
        # Initialize population
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        # Initialize mutation step sizes and concatenate
        pop = np.concatenate(((pop, np.random.uniform(step_l, step_u, (npop, n_vars)))), axis=1)

        # Evaluate initial fitness
        fit_pop = evaluate(pop[:, :n_vars])
        best = np.argmax(fit_pop) # index of best fitness value
        mean = np.mean(fit_pop) # mean fitness of population
        ini_g = 0
        solutions = [pop[:, :n_vars], fit_pop]
        env.update_solutions(solutions)

        # save first generation in df
        df = pd.DataFrame({'Exp_name':[experiment_name], 'Enemy':[enemy_lst], 'Run':[run],
                            'Gen':[0], 'Mean_fit':[mean], 'Max_fit':[fit_pop[best]]})

        # save df in csv file
        if not os.path.exists(experiment_name+'_results.csv'):
            df.to_csv(experiment_name+'_results.csv', index=False)
        else:
            append_list_as_row(experiment_name+'_results.csv', df.values.tolist()[0])

        # store max fitness value and accompanying weights
        max_fit = fit_pop[best] # store max fitness in the run
        best_weights = pop[best, :n_vars]
        
        for gen in range(ini_g+1, ngens):
            
            # create offspring
            offspring = blx(pop)
            fit_offspring = evaluate(offspring[:, :n_vars]) # evaluation

            # Elitism
            # find parents with highest fitness to safeguard for next gen
            top_idx = np.argpartition(fit_pop, -top_nr)[-top_nr:]
            top_parents = pop[top_idx]
            top_fit = fit_pop[top_idx]

            # find parents that are kept for next gen
            keep_idx = np.random.choice(pop.shape[0], offspring.shape[0], replace=False)
            
            # store safeguarded Elite and randomly selected parents in pop
            pop = np.vstack((pop[keep_idx], top_parents))
            fit_pop = np.append(fit_pop[keep_idx], top_fit)

            # include all offspring in pop
            pop = np.vstack((pop, offspring))
            fit_pop = np.append(fit_pop, fit_offspring)

            # evaluation of this gen
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)

            # check if max fitness was improved
            if max_fit < fit_pop[best]:
                best_weights = pop[best, :n_vars]

            # store generation in csv
            append_list_as_row(experiment_name+'_results.csv',
                                [experiment_name, enemy_lst, run, gen, mean, fit_pop[best]])
            

            # saves simulation state
            solutions = [pop[:, :n_vars], fit_pop]
            env.update_solutions(solutions)
            env.save_state()
        
        # store best weights of run in csv
        best_weights = np.append(best_weights, enemy_lst)

        if not os.path.exists(experiment_name+'_best.csv'):
            df_best_weights = pd.DataFrame({i:[j] for i,j in zip(range(n_vars+1), best_weights)})
            df_best_weights = df_best_weights.rename({n_vars:'Enemy'}, axis='columns')
            df_best_weights.to_csv(experiment_name+'_best.csv', index=False)
        else:
            append_list_as_row(experiment_name+'_best.csv', best_weights)


fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')



