################################
### test file Lara
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
import neat 

def eval_genomes():
    # takes as input the current population and the active configuration
    pass

def run(config_file):
    
    # load NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

    # create the population, which is the top-level object for a NEAT run
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 20)
    print(winner)


if __name__ == "__main__":
    experiment_name = 'lara_test'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
 
    for en in range(1, 9):
        env = Environment(experiment_name=experiment_name,
                        enemymode='static',
                        speed="normal",
                        sound="on",
                        fullscreen=True,
                        use_joystick=False,
                        playermode='ai')
        env.play()
        
    config_file = "config.txt"
    run(config_file)
