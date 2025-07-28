from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils
from open_spiel.python.egt import heuristic_payoff_table
from open_spiel.python.egt.examples import alpharank_example
#import heuristic_payoff_table as hpt
import pyspiel
from absl import app

import random
import numpy as np
from absl import logging 
import math
import copy
import multiprocessing as mp

import itertools
from scipy.spatial import distance
#from open_spiel.python.algorithms.adidas_utils.helpers import misc
#from open_spiel.python.algorithms.adidas_utils.solvers.nonsymmetric import adam

def sweep_pi_vs_alpha(payoff_tables,
                      strat_labels=None,
                      warm_start_alpha=None,
                      return_alpha=True,
                      m=50,
                      rtol=1e-5,
                      atol=1e-8,
                      num_strats_to_label=10,
                      legend_sort_clusters=False):
  """Computes stationary distribution, pi, for range of selection intensities.

  The range of selection intensities is defined in alpha_list and corresponds
  to the temperature of the Fermi selection function.

  Args:
    payoff_tables: List of game payoff tables, one for each agent identity. Each
      payoff_table may be either a numpy array, or a _PayoffTableInterface
      object.
    strat_labels: Human-readable strategy labels. See get_strat_profile_labels()
      in utils.py for formatting details.
    warm_start_alpha: Initial value of alpha to use.
    return_alpha: Whether to return the final alpha used.
    m: AlphaRank population size.
    rtol: The relative tolerance parameter for np.allclose calls.
    atol: The absolute tolerance parameter for np.allclose calls.
    num_strats_to_label: Number of strats to label in legend
    legend_sort_clusters: If true, strategies in the same cluster are sorted in
      the legend according to orderings for earlier alpha values. Primarily for
      visualization purposes! Rankings for lower alpha values should be
      interpreted carefully.

  Returns:
   pi: AlphaRank stationary distribution.
   alpha: The AlphaRank selection-intensity level resulting from sweep.
  """

  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  num_populations = len(payoff_tables)
  num_strats_per_population = utils.get_num_strats_per_population(
      payoff_tables, payoffs_are_hpt_format)

  if num_populations == 1:
    num_profiles = num_strats_per_population[0]
  else:
    num_profiles = utils.get_num_profiles(num_strats_per_population)

  assert (strat_labels is None or isinstance(strat_labels, dict)
          or (len(strat_labels) == num_profiles))

  pi_list = np.empty((num_profiles, 0))
  alpha_list = []
  num_iters = 0
  alpha_mult_factor = 2.

  if warm_start_alpha is not None:
    alpha = warm_start_alpha
    alpharank_succeeded_once = False
  else:
    alpha = 1e0  # Reasonable default for most games, can be user-overridden

  while 1:
    try:
      _, _, pi, _, _ = alpharank.compute(payoff_tables, alpha=alpha, m=m)
      pi_list = np.append(pi_list, np.reshape(pi, (-1, 1)), axis=1)
      alpha_list.append(alpha)
      # Stop when pi converges
      if num_iters > 0 and np.allclose(pi, pi_list[:, num_iters - 1], rtol,
                                       atol):
        break
      alpha *= alpha_mult_factor
      num_iters += 1
      alpharank_succeeded_once = True
    except ValueError as _:
      if warm_start_alpha is not None and not alpharank_succeeded_once:
        # When warm_start_alpha is used, there's a chance that
        # the initial warm_start_alpha is too large and causes exceptions due to
        # the Markov transition matrix being reducible. So keep decreasing until
        # a single success occurs.
        alpha /= 2
      elif not np.allclose(pi_list[:, -1], pi_list[:, -2], rtol, atol):
        # Sweep stopped due to multiple stationary distributions, but pi had
        # not converged due to the alpha scaling being too large.
        alpha /= alpha_mult_factor
        alpha_mult_factor = (alpha_mult_factor + 1.) / 2.
        alpha *= alpha_mult_factor
      else:
        break

  if return_alpha:
    return pi, alpha
  else:
    return pi
  
def get_top_ranking_agent(payoff_tables, pi, strat_labels):
  num_populations = len(payoff_tables)
  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  num_strats_per_population = utils.get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format)
  
  # Find the strategy with the maximum mass
  strat, mass = max(((strat, mass) for strat, mass in enumerate(pi)), key=lambda x: x[1], default=(None, None))
  
  if strat is not None:
      rounded_pi = np.round(mass, decimals=6)
      if num_populations == 1:
          strat_profile = strat
      else:
          strat_profile = utils.get_strat_profile_from_id(num_strats_per_population, strat)
      label = utils.get_label_from_strat_profile(num_populations, strat_profile, strat_labels)
      return label, np.abs(rounded_pi)
  
  return None, None


# Initialize firms
def initialize_firms(num_dirty, num_low_green, num_high_green, ka, ks, beta, lambdas):
    firms_dirty = [{'firm_type': 'dirty', 'ka': ka, 'ks': ks, 'market_share': 0, 'profit': 5,
                    'transformation_cost': 0, 'beta': beta, 'lambda_val': lambdas['dirty'], 'strategy': 'maintain'} 
                   for _ in range(num_dirty)]
    firms_low_green = [{'firm_type': 'low_green', 'ka': ka, 'ks': ks, 'market_share': 0, 'profit': 5,
                        'transformation_cost': 0, 'beta': beta, 'lambda_val': lambdas['low_green'], 'strategy': 'maintain'} 
                       for _ in range(num_low_green)]
    firms_high_green = [{'firm_type': 'high_green', 'ka': ka, 'ks': ks, 'market_share': 0, 'profit': 5,
                         'transformation_cost': 0, 'beta': beta, 'lambda_val': lambdas['high_green'], 'strategy': 'maintain'} 
                        for _ in range(num_high_green)]
    return firms_dirty, firms_low_green, firms_high_green

# Apply strategy
def apply_strategy(firm, strategy, lambdas):
    firm['strategy'] = strategy
    if strategy == 'upgrade':
        if firm['firm_type'] == 'dirty':
            firm['transformation_cost'] = firm['ka']
            firm['firm_type'] = 'low_green'
            firm['lambda_val'] = lambdas['low_green']  # Update lambda_val
        elif firm['firm_type'] == 'low_green':
            firm['transformation_cost'] = firm['ks']
            firm['firm_type'] = 'high_green'
            firm['lambda_val'] = lambdas['high_green']  # Update lambda_val
    else:
        firm['transformation_cost'] = 0


# Calculate payoff
def calculate_payoff(firm, total_subsidy, market_share):
    epsilon = 0  # Random disturbance removed here
    disturbed_profit = firm['profit'] * (1 + epsilon)
    current_profit = disturbed_profit + (market_share * total_subsidy) - firm['transformation_cost']
    discounted_profit = current_profit * firm['beta']
    return discounted_profit

# Run one round of simulation
def run_simulation(firms_dirty, firms_low_green, firms_high_green, num_dirty_upgrades, num_low_green_upgrades, total_subsidy, lambdas):
    upgrade_indices_dirty = random.sample(range(len(firms_dirty)), num_dirty_upgrades)
    for i, firm in enumerate(firms_dirty):
        if i in upgrade_indices_dirty:
            apply_strategy(firm, 'upgrade', lambdas)  # Pass lambdas
        else:
            apply_strategy(firm, 'maintain', lambdas)  # Pass lambdas

    upgrade_indices_low_green = random.sample(range(len(firms_low_green)), num_low_green_upgrades)
    for i, firm in enumerate(firms_low_green):
        if i in upgrade_indices_low_green:
            apply_strategy(firm, 'upgrade', lambdas)  # Pass lambdas
        else:
            apply_strategy(firm, 'maintain', lambdas)  # Pass lambdas

    
    

    num_high_green = sum(1 for firm in firms_low_green if firm['firm_type'] == 'high_green') + len(firms_high_green)
    num_low_green = sum(1 for firm in firms_low_green if firm['firm_type'] == 'low_green') + sum(1 for firm in firms_dirty if firm['firm_type'] == 'low_green')
    num_dirty = sum(1 for firm in firms_dirty if firm['firm_type'] == 'dirty')

    total_lambda_numerator = (num_high_green * lambdas['high_green'] + 
                              num_low_green * lambdas['low_green'] + 
                              num_dirty * lambdas['dirty'])
    total_lambda_denominator = sum(firm['lambda_val'] for firm in firms_dirty + firms_low_green + firms_high_green)

    for firm in firms_dirty + firms_low_green + firms_high_green:
        firm['market_share'] = (firm['lambda_val'] / total_lambda_numerator) * (total_subsidy / total_lambda_denominator)

    total_payoffs_dirty = [calculate_payoff(firm, total_subsidy, firm['market_share']) for firm in firms_dirty]
    #print(total_payoffs_dirty)
    total_payoffs_low_green = [calculate_payoff(firm, total_subsidy, firm['market_share']) for firm in firms_low_green]
    total_payoffs_high_green = [calculate_payoff(firm, total_subsidy, firm['market_share']) for firm in firms_high_green]

    avg_payoff_dirty = float(np.mean(total_payoffs_dirty)) if total_payoffs_dirty else 0
    #print(avg_payoff_dirty)
    avg_payoff_low_green = float(np.mean(total_payoffs_low_green)) if total_payoffs_low_green else 0
    avg_payoff_high_green = float(np.mean(total_payoffs_high_green)) if total_payoffs_high_green else 0

    return [avg_payoff_dirty, avg_payoff_low_green]


# Generate payoff matrix
def generate_payoff_matrix1(num_dirty, num_low_green, num_high_green, ka, ks, beta, total_subsidy, lambdas, num_simulations=10000):
    # Initialize firms, initial firm states are fixed
    initial_firms_dirty, initial_firms_low_green, initial_firms_high_green = initialize_firms(num_dirty, num_low_green, num_high_green, ka, ks, beta, lambdas)
    
    strat_dims = (num_dirty + 1, num_low_green + 1)
    payoff_matrix_sum = np.zeros((2,) + strat_dims)

    for sim in range(num_simulations):
        for num_dirty_upgrades in range(num_dirty + 1):
            for num_low_green_upgrades in range(num_low_green + 1):
                # Deep copy initial firms to ensure each simulation uses the same initial state
                firms_dirty = copy.deepcopy(initial_firms_dirty)
                firms_low_green = copy.deepcopy(initial_firms_low_green)
                firms_high_green = copy.deepcopy(initial_firms_high_green)
                
                # Run simulation
                avg_payoffs = run_simulation(firms_dirty, firms_low_green, firms_high_green, num_dirty_upgrades, num_low_green_upgrades, total_subsidy, lambdas)
                #print(avg_payoffs)
                
                # Accumulate payoffs into matrix
                payoff_matrix_sum[0, num_dirty_upgrades, num_low_green_upgrades] += avg_payoffs[0]
                payoff_matrix_sum[1, num_dirty_upgrades, num_low_green_upgrades] += avg_payoffs[1]

    payoff_matrix_avg = payoff_matrix_sum / num_simulations #If average is needed, uncomment this line
    return payoff_matrix_avg



num_dirty = 50
num_low_green = 50
num_high_green = 50  # Number of high green firms

beta = 0.9  # Time discount factor
total_subsidy = 1500  # Total government subsidy

def adjust_lambdas(adjust_factor):
    base_dirty = 10  # Initial value of λ['dirty']
    lambdas = {
        'dirty': base_dirty,
        'low_green': base_dirty + adjust_factor * 10,   # Difference from dirty: 10 * h
        'high_green': base_dirty + adjust_factor * 20   # Difference from dirty: 20 * h
    }
    return lambdas

lambdas_factor = 1
lambdas_step = 1
lambdas = adjust_lambdas(lambdas_factor)

k = 1
k_step = 1
ka = k * lambdas['dirty']
ks = k * lambdas['low_green']



def experiment_task(v, lambdas_factor, lambdas_step, k, k_step, num_dirty, num_low_green, num_high_green, beta, total_subsidy):
    # Adjust lambdas
    f = lambdas_factor + v * lambdas_step  # Calculate lambdas_factor based on v
    lambdas = adjust_lambdas(f)
    
    inner_results = []
    for i in range(10):
        ka = [k + j * k_step for j in range(10)][i]
        ks = [k + j * k_step for j in range(10)][i]
        try:
            # Generate payoff matrix and calculate
            payoff = generate_payoff_matrix1(num_dirty, num_low_green, num_high_green, ka, ks, beta, total_subsidy, lambdas)
            pi, alpha = sweep_pi_vs_alpha(payoff)
            #print(alpha)
            payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff)
            strat_labels = utils.get_strat_profile_labels(payoff, payoffs_are_hpt_format)
            agent_set, score = get_top_ranking_agent(payoff, pi, strat_labels)
            inner_results.append(agent_set)
        except Exception as e:
            print(f"Error at lambda['dirty'] = {f}, ka = {ka}: {e}")
            inner_results.append("error")
    
    return v, inner_results  # Return v to ensure result order

if __name__ == "__main__":
    # Define list to store results
    experiment_results = [None] * 10  # Pre-define list of size 10 to ensure order

    # Create multiprocessing pool
    cpu_count = 10  # Or manually specify like 4 cores: mp.Pool(4)
    pool = mp.Pool(cpu_count)

    # Execute outer loop in parallel
    results = [pool.apply_async(experiment_task, args=(v, lambdas_factor, lambdas_step, k, k_step, num_dirty, num_low_green, num_high_green, beta, total_subsidy)) for v in range(10)]

    # Get results from each process and store in experiment_results according to outer loop order
    for res in results:
        v, inner_results = res.get()  # Get v and inner_results returned by multiprocessing
        experiment_results[v] = inner_results  # Save results in order of v

    # Close process pool
    pool.close()
    pool.join()

    # Define the file path
    file_path = '/scratch/users/Green Disclosure/result/experiment_results_lax.txt'

    # Write the results to a file
    with open(file_path, 'w') as file:
        for result in experiment_results:
            file.write(", ".join(result) + "\n")

