"""
swarmSimModel.py - 2020 Aug 13
 - stability factor for reducing 'hunting': a threshhold below which an agent's mvt in current step is clamped to 0
 - Perimeter-packing code enhanced: effective repulsion between neighbours on perimeter is reduced:
     both effective repulsion radius and repulsion weighting (in computation of resultant movement) are reduced by
     p-p factor, a fractional number between 0 and 1. At the same time, cohesion weighting between these pairs is
     boosted by the reciprocal of the p-p factor.
 - It has been observed that agents near/on perim often hop on and off the perim too often. The tweaks just mentioned
     are motivated partly by this. Also, the perimeter determining function function onPerim has been tweaked to
     make it more likely neibouring perimeter agents will stay on perimeter.
 - The d_step function has been decomposed into several subfunctions: all_pairs_mag, compute_coh, nbr_sort, (helper to)
     onPerim, compute_erf (tweaks repulsion radius and coh, rep wgts between perimeter agents),
     compute_rep_linear, compute_rep_quadratic, compute_rep_exponential, update_resultant (applies stability factor)
 - d-step creates array ecb by broadcasting cohesion radii to pairs of agents, then initialises other arrays, updates
     with all_pairs_mag(). It then updates perimeter status of agents, computes effective repulsion and updates
     resultant movement.
 - d-step and all its helpers are decorated with numba @jit decorations to boost performance
"""
import json
import numpy as np
from numba import jit, prange

# Define some useful array accessor constants
POS_X  = 0    # x-coordinates of agents position
POS_Y  = 1    # y-coordinates of agents position
COH_X  = 2    # x-coordinates of cohesion vectors
COH_Y  = 3    # y-coordinates of cohesion vectors
REP_X  = 4    # x-coordinates of repulsion vectors
REP_Y  = 5    # y-coordinates of repulsion vectors
DIR_X  = 6    # x-coordinates of direction vectors
DIR_Y  = 7    # y-coordinates of direction vectors
RES_X  = 8    # x-coordinates of resultant vectors
RES_Y  = 9    # y-coordinates of resultant vectors
GOAL_X = 10   # x-coordinates of goals
GOAL_Y = 11   # y-coordinates of goals
CF     = 12   # cohesion field radii
RF     = 13   # repulsion field radii
KC     = 14   # cohesion vector scaling factor
KR     = 15   # repulsion vector scaling factor
KD     = 16   # direction vector scaling factor
PRM    = 17   # if True agent known to be on perimeter of swarm
COH_N  = 18   # number of cohesion neighbours
REP_N  = 19   # number of repulsion neighbours

N_ROWS = 20   # number of rows in array that models swarm state
eps    = np.finfo('float64').eps # smallest positive 64 bit float value

default_swarm_params = {
    'cb' : 4.0,
    'rb' : 3.0,
    'ob' : 3.0,
    'kc' : 1.0,
    'kr' : 1.0,
    'kd' : 0.0,
    'ko' : 0.0,
    'scaling' : 'linear',
    'exp_rate' : 0.2,
    'speed' : 0.05,
    'stability_factor' : 0.0,
    'pc' : 1.0,
    'pr' : 1.0
}

def mk_rand_swarm(n, *, cb=4.0, rb=3.0, kc=1.0, kr=1.0, kd=0.0, goal=[[0.0], [0.0]], loc=0.0, grid=10, seed=None):
    '''
    create a 2-D array of N_ROWS attributes for n agents.

    :param n:      number of agents
    :param cb:     cohesion field radius of all agents; default 4.0; heterogeneous fields are allowed but not catered for here
    :param rb:     repulsion field radius of all agents; default 3.0
    :param kc:     weighting factor for cohesion component, default 1.0
    :param kr:     weighting factor for repulsion component, default 1.0
    :param kd:     weighting factor for direction component, default 0.0 (i.e. goal is ignored by default)
    :param goal:   location of a goal for all agents; heterogeneous goals are allowed but not catered for here
    :param loc:    location of agent b_0 -- the focus of the swarm
    :param grid:   size of grid around b_0 in which all other agents will be placed initially at random
    '''
    b = np.empty((N_ROWS, n))                       #create a 2-D array, big enough for n agents
    prng = np.random.default_rng(seed)
    np.copyto(b[POS_X:POS_Y + 1,:], (prng.random(size=2 * n) * 2 * grid - grid + loc).reshape(2, n)) # place agents randomly
    b[POS_X:POS_Y + 1,0] = loc                      # b_0 placed at [loc, loc]
    b[COH_X:COH_Y+1,:] = 0.                         # cohesion vectors initially [0.0, 0.0]
    b[REP_X:REP_Y+1,:] = 0.                         # repulsion vectors initially [0.0, 0.0]
    b[DIR_X:DIR_Y+1,:] = 0.                         # direction vectors initially [0.0, 0.0]
    b[RES_X:RES_Y + 1,:] = 0.                       # resultant vectors initially [0.0, 0.0]
    b[GOAL_X:GOAL_Y + 1,:] = goal                   # goal is at [goal[0], goal[1]], default [0.0, 0.0]
    print(f"Goal is {b[GOAL_X:GOAL_Y + 1,:]}") 
    b[CF,:] = cb                                    # cohesion field of all agents set to cb
    b[RF,:] = rb                                    # repulsion field of all agents set to rb
    b[KC,:] = kc                                    # cohesion weight for all agents set to kc
    b[KR,:] = kr                                    # repulsion weight for all agents set to kr
    b[KD,:] = kd                                    # direction weight for all agents set to kd
    b[PRM,:] = False                                # initially no agents known to be on perimeter
    b[COH_N,:] = 0.                                 # initially no cohesion neighbours
    b[REP_N,:] = 0.                                 # initially no repulsion neighbours
    return b

def mk_swarm(xs, ys, *, cb=4.0, rb=3.0, kc=1.0, kr=1.0, kd=0.0, goal=[[0.0],[0.0]]):
    '''
    create a 2-D array of N_ROWS attributes for len(xs) agents.

    :param xs:      x-values of position of agents
    :param ys:      y-values of position of agents
    :param cb:      cohesion field radius of all agents; default 4.0; heterogeneous fields are allowed but not catered for here
    :param rb:      repulsion field radius of all agents; default 3.0
    :param kc:      weighting factor for cohesion component, default 1.0
    :param kr:      weighting factor for repulsion component, default 1.0
    :param kd:      weighting factor for direction component, default 0.0 (i.e. goal is ignored by default)
    :param goal:    location of a goal for all agents; heterogeneous goals are allowed but not catered for here
    '''
    n = len(xs)
    assert len(ys) == n
    b = np.empty((N_ROWS, n))                       # create a 2-D array, big enough for n agents
    np.copyto(b[POS_X], xs)                         # place agents as specified
    np.copyto(b[POS_Y], ys)                         # place agents as specified
    b[COH_X:COH_Y+1,:] = 0.                         # cohesion vectors initially [0.0, 0.0]
    b[REP_X:REP_Y+1,:] = 0.                         # repulsion vectors initially [0.0, 0.0]
    b[DIR_X:DIR_Y+1,:] = 0.                         # direction vectors initially [0.0, 0.0]
    b[RES_X:RES_Y + 1,:] = 0.                       # resultant vectors initially [0.0, 0.0]
    b[GOAL_X:GOAL_Y + 1,:] = goal                   # goal is at [goal, goal], default [0.0, 0.0]
    b[CF,:] = cb                                    # cohesion field of all agents set to cb
    b[RF,:] = rb                                    # repulsion field of all agents set to rb
    b[KC,:] = kc                                    # cohesion weight for all agents set to kc
    b[KR,:] = kr                                    # repulsion weight for all agents set to kr
    b[KD,:] = kd                                    # direction weight for all agents set to kd
    b[PRM,:] = False                                # initially no agents known to be on perimeter
    b[COH_N,:] = 0.                                 # initially no cohesion neighbours
    b[REP_N,:] = 0.                                 # initially no repulsion neighbours
    return b

@jit(nopython=True, fastmath=True, cache=True)
def all_pairs_mag(b, xv, yv, mag, ecb):
    n_agents = b.shape[1]
    b[COH_N].fill(0.)
    for i in range(n_agents):
        for j in range(i):
            xv[i,j] = b[POS_X][i] - b[POS_X][j]
            xv[j,i] = -xv[i,j]
            yv[i,j] = b[POS_Y][i] - b[POS_Y][j]
            yv[j,i] = -yv[i,j]
            mag[i,j] = np.sqrt(xv[i,j] ** 2 + yv[i,j] ** 2)
            mag[j,i] = mag[i,j]
            if mag[j,i] <= ecb[j,i]:
                b[COH_N][i] = b[COH_N][i] + 1
                b[COH_N][j] = b[COH_N][j] + 1
        xv[i,i] = 0.0
        yv[i,i] = 0.0
        mag[i,i] = 0.0

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def compute_coh(b, xv, yv, mag, ecb, ekc):
    n_agents = b.shape[1]
    for i in prange(n_agents):
        b[COH_X][i] = 0.0
        b[COH_Y][i] = 0.0
        for j in range(n_agents):
            if j != i and mag[j, i] <= ecb[j, i]:
                b[COH_X][i] = b[COH_X][i] + (xv[j,i] * ekc[j,i])
                b[COH_Y][i] = b[COH_Y][i] + (yv[j,i] * ekc[j,i])

@jit(nopython=True, fastmath=True, cache=True)
def nbr_sort(a, ang, i):
    n = a.shape[0]
    for j in range(n):
        jmin = j
        for k in range(j, n):
            if (ang[:,i][a[k]] < ang[:,i][a[jmin]]):
                jmin = k
        if jmin != j:
            a[jmin], a[j] = a[j], a[jmin]

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def onPerim(b, xv, yv, mag, ecb):
    n_agents = b.shape[1]
    result = np.full(n_agents, False)
    ang = np.arctan2(yv, xv)                    # all pairs polar angles
    for i in prange(n_agents):
        if b[COH_N][i] < 3:
            result[i] = True
            continue
        nbrs = np.full(int(b[COH_N][i]), 0)
        k = 0
        for j in range(n_agents):
            if j != i and mag[j, i] <= ecb[j, i]:
                nbrs[k] = j
                k += 1
        nbr_sort(nbrs, ang, i)
        for j in range(int(b[COH_N][i])):
            k = (j + 1) % int(b[COH_N][i])
            if mag[nbrs[k],nbrs[j]] > ecb[nbrs[k],nbrs[j]]: # nbrs[j] and nbrs[k] are not cohesion neighbours
                result[i] = True
                break
            delta = ang[:,i][nbrs[k]] - ang[:,i][nbrs[j]]
            if (delta < 0):
                delta += np.pi * 2.0;
            if (delta > np.pi) or (b[PRM][i] and delta > 2.8):
                result[i] = True;
                break;
    return result, ang

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def compute_erf(b, cscale, rscale):
    n_agents = b.shape[1]
    erf = np.empty((n_agents, n_agents))
    ekc = np.empty((n_agents, n_agents))
    ekr = np.empty((n_agents, n_agents))
    for i in prange(n_agents):
        for j in range(i + 1):
            if b[PRM][i] and b[PRM][j]:
                erf[i,j] = b[RF][i] * rscale
                erf[j,i] = b[RF][j] * rscale
                # ekc[i,j] = b[KC][i] * (1. / scale)
                # ekc[j,i] = b[KC][j] * (1. / scale)
                ekc[i,j] = b[KC][i] * cscale
                ekc[j,i] = b[KC][j] * cscale
                # ekr[i,j] = b[KR][i] * scale
                # ekr[j,i] = b[KR][j] * scale
                ekr[i,j] = b[KR][i]
                ekr[j,i] = b[KR][j]
            else:
                erf[i,j] = b[RF][i]
                erf[j,i] = b[RF][j]
                ekc[i,j] = b[KC][i]
                ekc[j,i] = b[KC][j]
                ekr[i,j] = b[KR][i]
                ekr[j,i] = b[KR][j]
    return erf, ekc, ekr

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def compute_rep_linear(b, xv, yv, mag, erf, ekr):
    n_agents = b.shape[1]
    for i in prange(n_agents):
        b[REP_N][i] = 0.0
        b[REP_X][i] = 0.0
        b[REP_Y][i] = 0.0
        for j in range(n_agents):
            if j != i and mag[j, i] <= erf[i,j]:
                b[REP_N][i] = b[REP_N][i] + 1
                # b[REP_X][i] = b[REP_X][i] + ((mag[j,i] - erf[i,j]) * (xv[j,i] / mag[j,i]) * ekr[j,i])
                # b[REP_Y][i] = b[REP_Y][i] + ((mag[j,i] - erf[i,j]) * (yv[j,i] / mag[j,i]) * ekr[j,i])
                b[REP_X][i] = b[REP_X][i] + (1 - (erf[i,j] / mag[j,i])) * xv[j,i] * ekr[j,i]
                b[REP_Y][i] = b[REP_Y][i] + (1 - (erf[i,j] / mag[j,i])) * yv[j,i] * ekr[j,i]

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def compute_rep_quadratic(b, xv, yv, mag, erf, ekr):
    n_agents = b.shape[1]
    for i in prange(n_agents):
        b[REP_N][i] = 0.0
        b[REP_X][i] = 0.0
        b[REP_Y][i] = 0.0
        for j in range(n_agents):
            if j != i and mag[j, i] <= erf[i,j]:
                b[REP_N][i] = b[REP_N][i] + 1
                b[REP_X][i] = b[REP_X][i] + (-erf[i,j] * (mag[j,i] ** -2) * (xv[j,i] / mag[j,i]) * ekr[j,i])
                b[REP_Y][i] = b[REP_Y][i] + (-erf[i,j] * (mag[j,i] ** -2) * (yv[j,i] / mag[j,i]) * ekr[j,i])

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def compute_rep_exponential(b, xv, yv, mag, erf, ekr, exp_rate):
    n_agents = b.shape[1]
    for i in prange(n_agents):
        b[REP_N][i] = 0.0
        b[REP_X][i] = 0.0
        b[REP_Y][i] = 0.0
        for j in range(n_agents):
            if j != i and mag[j, i] <= erf[i,j]:
                b[REP_N][i] = b[REP_N][i] + 1
                b[REP_X][i] = b[REP_X][i] + (-erf[i,j] * (np.e ** (-mag[j,i] * exp_rate)) * (xv[j,i] / mag[j,i]) * ekr[j,i])
                b[REP_Y][i] = b[REP_Y][i] + (-erf[i,j] * (np.e ** (-mag[j,i] * exp_rate)) * (yv[j,i] / mag[j,i]) * ekr[j,i])

@jit(nopython=True, fastmath=True, cache=True)
def update_resultant(b, stability_factor, speed):
    n_agents = b.shape[1]
    for i in range(n_agents):
        mag_res = np.sqrt(b[RES_X][i] ** 2 + b[RES_Y][i] ** 2)
        if mag_res > stability_factor * speed:
            b[RES_X][i] = b[RES_X][i] / mag_res * speed
            b[RES_Y][i] = b[RES_Y][i] / mag_res * speed
        else:
            b[RES_X][i] = 0.0
            b[RES_Y][i] = 0.0

def compute_step(b, *, scaling='linear', exp_rate=0.2, speed=0.05, perim_coord=False, stability_factor=0.0, pc=1.0, pr=1.0):
    """
    Compute one step in the evolution of swarm `b`, update the COH, REP, DIR and RES fields
    :param b: the array modelling the state of the swarm
    :param scaling: choose 'linear', 'quadratic', or 'exponential' scaling of repulsion vectors
    :param exp_rate: rate of scaling in 'exponential' case
    :param speed: the speed of each agent, i.e. the number of simulation distance units per simulation time unit (step)
    :param stability_factor: if the magnitude of an agent's resultant vector is less than speed * stability_factor then agent does not move
    :param pc: determines the amount by which the cohesion weight should be increased for perimeter agents
                                     e.g. a pc of 2.0 causes the size of the cohesion weight to be doubled
    :param pr: determines the amount by which the repulsion field should be reduced for perimeter agents,
                                     e.g. a pr of 0.5 causes the size of the repulsion field to be halved
    """
    # print(scaling, exp_rate, speed, perim_coord, stability_factor, pr)
    n_agents = b.shape[1]
    ecb = np.broadcast_to(b[CF], (b[CF].shape[0], b[CF].shape[0]))
    xv = np.empty((n_agents, n_agents))
    yv = np.empty((n_agents, n_agents))
    mag = np.empty((n_agents, n_agents))
    all_pairs_mag(b, xv, yv, mag, ecb)

    # compute the perimeter
    b[PRM], ang = onPerim(b, xv, yv, mag, ecb)

   # compute the effective repulsion field, cohesion weight and repulsion weight
    erf, ekc, ekr = compute_erf(b, pc, pr)

    # compute the cohesion vectors
    compute_coh(b, xv, yv, mag, ecb, ekc)
    b[COH_X:COH_Y+1] /= np.maximum(b[COH_N], 1)         # divide by the number of cohesion neighbours

     # compute the repulsion vectors
    if scaling == 'linear':
        compute_rep_linear(b, xv, yv, mag, erf, ekr)
    elif scaling == 'quadratic':
        compute_rep_quadratic(b, xv, yv, mag, erf, ekr)
    elif scaling == 'exponential':
        compute_rep_exponential(b, xv, yv, mag, erf, ekr, exp_rate)
    else:
        assert(False)                                   # something's gone wrong here
    b[REP_X:REP_Y+1] /= np.maximum(b[REP_N], 1)         # divide by the number of repulsion neighbours

    # compute the direction vectors
    b[DIR_X:DIR_Y+1] = b[KD] * (b[GOAL_X:GOAL_Y+1] - b[POS_X:POS_Y+1])

    # compute the resultant of the cohesion, repulsion and direction vectors
    if perim_coord:
        b[RES_X:RES_Y+1] = b[COH_X:COH_Y+1] + b[REP_X:REP_Y+1] + b[PRM] * b[DIR_X:DIR_Y+1]
    else:
        b[RES_X:RES_Y+1] = b[COH_X:COH_Y+1] + b[REP_X:REP_Y+1] + b[DIR_X:DIR_Y+1]

    # normalise the resultant and update for speed, adjusted for stability
    update_resultant(b, stability_factor, speed)

    return xv, yv, mag, ang, ecb, erf, ekc, ekr         # helpful in calculation of metrics, instrumentation, debugging

def apply_step(b):
    """
    Assuming the step has been computed so that RES fields are up to date, update positions
    """
    b[POS_X:POS_Y+1] += b[RES_X:RES_Y+1]

@jit(nopython=True, fastmath=True)
def mu_sigma_d(mag, ecb):
    n_agents = mag.shape[0]
    msum = 0; msum_sq = 0; nsum = 0
    for i in prange(n_agents):
        for j in range(i):
            if mag[j, i] <= ecb[j, i]:
                msum += mag[j, i]
                msum_sq += mag[j, i] **2
                nsum += 1
            if mag[i, j] <= ecb[i, j]:
                msum += mag[i, j]
                msum_sq += mag[i, j] **2
                nsum += 1
    mu_d = msum / nsum
    mu_d_sq = msum_sq / nsum
    var_d = mu_d_sq - mu_d ** 2
    sigma_d = np.sqrt(var_d)
    return mu_d, sigma_d

def mu_sigma_p(b):
    vcr_x = b[COH_X] + b[REP_X]                                 # the weighted cohesion/repulsion vector of every agent
    vcr_y = b[COH_Y] + b[REP_Y]
    vcr_mag = np.hypot(vcr_x, vcr_y)                            # the magnitude of the weighted cohesion/repulsion vector of every agent
    vc_mag = np.hypot(b[COH_X], b[COH_Y])                       # the magnitude of the cohesion component of the cohesion/repulsion vector
    vr_mag = np.hypot(b[REP_X], b[REP_Y])                       # the magnitude of the repulsion component of the cohesion/repulsion vector
    P = np.where(vc_mag > vr_mag, vcr_mag, -vcr_mag)            # the implementation of P as defined
    n_agents = b.shape[1]                                       # the total number of agents in the swarm
    mu_p = np.sum(P) / n_agents                                 # the mean
    sigma_p = np.sqrt(np.sum((P - mu_p) ** 2) / n_agents)       # the standard deviation
    return mu_p, sigma_p

'''
Data persistence methods
'''

def saveState(b, path):
    """
    Save state of a swarm model
    :b: numpy array representing state of a swarm
    :path: path to a file to which the data are to be saved
    """
    with open(path, 'wt') as f:
        for n in range(np.ma.size(b,1)):
          for r in range(np.ma.size(b,0)):
              f.write("{:f}\t".format(b[r][n]))
          f.write("\n")
        f.close()
    print("{:d} agents saved.".format(np.ma.size(b,1)))

def loadState(path):
    """
    Load state of a swarm model from saved data
    :b: numpy array representing state of a swarm
    :path: path to a file from which the data are to be loaded
    """
    with open(path, 'rt') as f:
      lines = f.readlines()
    f.close()
    print("{:d} lines read.".format(len(lines)))
    nums = [[float(x) for x in line.split()] for line in lines]
    return np.transpose(np.array(nums))

def readCoords(path):
    """
    Read a set of coordinates for agents from a text file of lines each
    containing an x- and a y- coordinate.
    Return two lists, xs, ys for use by make_swarm(...) function
    :path: path to a file from which the data are to be loaded
    """
    with open(path, 'rt') as f:
      lines = f.readlines()
    f.close()
    cds = []
    for ln in lines:
      for wd in ln.split():
        cds.append(float(wd))
    xs = cds[0::2]
    ys = cds[1::2]
    return xs, ys

def dump_swarm(b, swarm_args, step_args):
    goal = swarm_args['goal']
    swarm_args = {k:v for k,v in swarm_args.items() if k in ['cb', 'rb', 'kc', 'kr', 'kd']}
    coords = b[POS_X:POS_Y+1,:].tolist()
    coords.append([0.0] * b.shape[1])
    state = {
        'params': {**default_swarm_params, **swarm_args, **step_args},
        'agents': {'coords': coords}, 
        'destinations' : {'coords': [goal[0], goal[1], [0.0]]},
        'obstacles' : {'coords': [[],[],[]]} 
    }
    with open('swarm.json', 'w') as f:
        json.dump(state, f, indent=2)
        f.close()

def load_swarm():
    with open('swarm.json', 'r') as f:
        state = json.load(f)
        f.close()
    return state
