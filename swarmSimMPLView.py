"""
swarmSimMPL.py
Matplotlib-based animated display for swarm simulator.
"""
import matplotlib.pyplot as plt
import swarmSimModel as mdl
import sys
import argparse

#Some boiler-plate and functions to assist with plotting and animation
plt.rc('font', family='serif', size=4)
plt.rc('figure', dpi=200)
plt.rc('axes', axisbelow=True, titlesize=5)
plt.rc('lines', linewidth=1)
plt.rcParams.update({'figure.max_open_warning': 0})
from matplotlib.animation import FuncAnimation
import numpy as np

def run_simulation(b, **kwargs):
    """
    run a simulation of stepping in a simple matplotlib graphical environment
    
    :param b: the array modelling the state of the swarm
    :param with_perimeter: if True, distinguish between perimeter and internal agents
    :param **kwargs: keyword arguments for the step function
    """
    fig, ax = plt.subplots(figsize=(4,4))       # create a graph
    mdl.compute_step(b, **kwargs)               # compute first step

    def simulate(i):
        """
        Ultra-simple simulation function  
        """
        ax.cla()                                                                  # clear the axes
        ax.set(xlim=(-10, 10), ylim=(-10, 10))                                    # set the limits of the axes
        mdl.apply_step(b)                                                         # apply step
        p = b[mdl.PRM] != 0; q = np.logical_not(p)                                # compute the perimeter
        snapshot = ax.plot(b[mdl.POS_X, p], b[mdl.POS_Y, p], 'ro',                # plot perimeter agents
                           b[mdl.POS_X, q], b[mdl.POS_Y, q], 'ko', markersize=2)  # plot internal agents
        mdl.compute_step(b, **kwargs)                                             # take next step
        return snapshot

    def init():
        return []
    
    # return a function that calls `simulate` every 100 ms and updates the figure
    return FuncAnimation(fig, simulate, interval=100, init_func=init)


'''
Run the MPLView
:args: a list of command-line arguments usually from sys.argv[1] onwards
(Python, like C(++) but unlike java, includes the command as argv[0].)
'''
def runMPLView(args):
  swarm_args = {k:v for k,v in args.items() if k in ['random', 'load_state', 'read_coords', 'cf', 'rf', 'kc', 'kr', 'kd', 'goal', 'loc', 'grid', 'seed'] and v is not None}
  step_args = {k:v for k,v in args.items() if k in ['scaling', 'exp_rate', 'speed', 'perimeter_directed', 'stability_factor', 'perimeter_packing_factor'] and v is not None} 
  if 'random' in swarm_args.keys():
    n = swarm_args['random']
    del swarm_args['random']
    b = mdl.mk_rand_swarm(n, **swarm_args)
  elif 'read_coords' in swarm_args.keys():
    xs, ys = mdl.readCoords(swarm_args['read_coords'])
    del swarm_args['read_coords']
    b = mdl.mk_swarm(xs, ys, **swarm_args)
  elif 'load_state' in swarm_args.keys():
    b = mdl.loadState(swarm_args['load_state'])
  else:
    print("Error in swarm creation")
    return

  print(step_args)
  sim = run_simulation(b, **step_args)
  plt.show()


################################ main line ########################################

parser = argparse.ArgumentParser()
swarm = parser.add_mutually_exclusive_group(required=True)
swarm.add_argument('-r', '--random', type=int, help='create random swarm of size RANDOM')
swarm.add_argument('-s', '--load_state', help='load initial swarm state from LOAD_STATE')
swarm.add_argument('-c', '--read_coords', help='read initial agent positions from READ_COORDS')
parser.add_argument('--cf', type=float, help='radius of the cohesion field')
parser.add_argument('--rf', type=float, help='radius of the repulsion field')
parser.add_argument('--kc', type=float, help='weight of the cohesion vector')
parser.add_argument('--kr', type=float, help='weight of the repulsion vector')
parser.add_argument('--kd', type=float, help='weight of the direction vector')
parser.add_argument('--goal', type=float, help='the swarm has a goal with coordinates (GOAL, GOAL)')
parser.add_argument('--loc', type=float, help='initially centre of the swarm is at coordinates (LOC, LOC)')
parser.add_argument('--grid', type=float, help='initially swarms is distributed in an area of 2.GRID x 2.GRID')
parser.add_argument('--seed', type=int, help='seed for random number generator for random swarm')
parser.add_argument('--scaling', choices=['linear', 'quadratic', 'exponential'], help='scaling method for computation of repulsion vector')
parser.add_argument('--exp_rate', type=float, help='exponential rate if scaling="exponential"')
parser.add_argument('--speed', type=float, help='distance moved per unit time')
parser.add_argument('--perimeter_directed', action='store_true', help='use only perimeter agents in goal seeking')
parser.add_argument('--stability_factor', type=float, help='constrain agent movement if magnitude of resultant vector is less than STABILITY_FACTOR * speed')
parser.add_argument('--perimeter_packing_factor', type=float, help='reduce repulsion field by PERIMETER_PACKING_FACTOR for perimeter agents')
args = vars(parser.parse_args())
runMPLView(args)

