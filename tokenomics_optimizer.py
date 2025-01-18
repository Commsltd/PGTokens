import numpy as np
from typing import Dict, List, Any
import pandas as pd
from scipy.optimize import differential_evolution
from simulation import simulate_tokenomics

def calculate_token_price_score(results):
    """Calculate score based on token price performance"""
    # Get price metrics
    final_price = results['Token Price'].iloc[-1]
    initial_price = results['Token Price'].iloc[0]
    price_change = (final_price - initial_price) / initial_price
    
    # Calculate volatility
    price_volatility = results['Token Price'].std() / results['Token Price'].mean()
    
    # Calculate price stability (penalize high volatility)
    stability_score = 1 / (1 + price_volatility)
    
    # Combine metrics with weights
    score = (
        0.4 * max(0, price_change) +  # 40% weight on price growth
        0.4 * stability_score +        # 40% weight on stability
        0.2 * (final_price / initial_price)  # 20% weight on final price ratio
    )
    
    return np.clip(score, 0, 1)

def calculate_user_growth_score(results):
    """Calculate score based on user growth metrics"""
    # Get user metrics
    final_users = results['Users'].iloc[-1]
    initial_users = results['Users'].iloc[0]
    user_growth = (final_users - initial_users) / initial_users
    
    # Calculate user retention
    retention_rate = 1 - results['Users'].diff().clip(upper=0).abs().sum() / results['Users'].sum()
    
    # Calculate premium adoption
    premium_ratio = results['Premium Users'].iloc[-1] / results['Users'].iloc[-1]
    
    # Combine metrics with weights
    score = (
        0.4 * min(1, user_growth) +     # 40% weight on growth
        0.3 * retention_rate +           # 30% weight on retention
        0.3 * premium_ratio             # 30% weight on premium adoption
    )
    
    return np.clip(score, 0, 1)

def calculate_reserve_health_score(results):
    """Calculate score based on reserve health metrics"""
    # Get reserve metrics
    final_reserve = results['Reward Pool'].iloc[-1]
    initial_reserve = results['Reward Pool'].iloc[0]
    reserve_ratio = final_reserve / initial_reserve
    
    # Calculate reserve stability
    reserve_volatility = results['Reward Pool'].std() / results['Reward Pool'].mean()
    stability_score = 1 / (1 + reserve_volatility)
    
    # Calculate minimum reserve ratio
    min_ratio = results['Reward Pool'].min() / initial_reserve
    
    # Combine metrics with weights
    score = (
        0.4 * min(1, reserve_ratio) +  # 40% weight on final ratio
        0.3 * stability_score +         # 30% weight on stability
        0.3 * min(1, min_ratio * 2)    # 30% weight on minimum ratio
    )
    
    return np.clip(score, 0, 1)

def calculate_token_velocity_score(results):
    """Calculate score based on token velocity metrics"""
    # Calculate token velocity (spending rate)
    velocity = results['Monthly Token Spending'].mean() / results['Tokens Earned'].mean()
    
    # Calculate staking ratio
    staking_ratio = results['Total Tokens Staked'].mean() / results['Tokens Earned'].mean()
    
    # Calculate spending stability
    spending_volatility = results['Monthly Token Spending'].std() / results['Monthly Token Spending'].mean()
    stability_score = 1 / (1 + spending_volatility)
    
    # Combine metrics with weights
    score = (
        0.4 * min(1, velocity) +       # 40% weight on velocity
        0.3 * staking_ratio +          # 30% weight on staking
        0.3 * stability_score          # 30% weight on stability
    )
    
    return np.clip(score, 0, 1)

def perform_sensitivity_analysis(history):
    """Analyze parameter sensitivity"""
    param_names = list(history[0]['params'].keys())
    sensitivities = {}
    
    for param in param_names:
        # Get unique parameter values and corresponding scores
        values = [h['params'][param] for h in history]
        scores = [h['weighted_score'] for h in history]
        
        # Calculate correlation
        correlation = np.corrcoef(values, scores)[0, 1]
        
        # Calculate average impact
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        impact = std_score / mean_score if mean_score > 0 else 0
        
        sensitivities[param] = {
            'correlation': correlation,
            'impact': impact,
            'importance': abs(correlation * impact)
        }
    
    return sensitivities

def analyze_trade_offs(history):
    """Analyze trade-offs between different objectives"""
    trade_offs = {}
    
    # Get all objective pairs
    objectives = list(history[0]['scores'].keys())
    for i, obj1 in enumerate(objectives):
        for obj2 in objectives[i+1:]:
            # Calculate correlation between objectives
            scores1 = [h['scores'][obj1] for h in history]
            scores2 = [h['scores'][obj2] for h in history]
            correlation = np.corrcoef(scores1, scores2)[0, 1]
            
            # Identify Pareto optimal solutions
            pareto_points = []
            for h in history:
                is_pareto = True
                for other in history:
                    if (other['scores'][obj1] > h['scores'][obj1] and 
                        other['scores'][obj2] > h['scores'][obj2]):
                        is_pareto = False
                        break
                if is_pareto:
                    pareto_points.append({
                        'params': h['params'],
                        'scores': {obj1: h['scores'][obj1], obj2: h['scores'][obj2]}
                    })
            
            trade_offs[f"{obj1}_vs_{obj2}"] = {
                'correlation': correlation,
                'pareto_points': pareto_points,
                'conflict_level': abs(correlation) if correlation < 0 else 0
            }
    
    return trade_offs

def run_tokenomics_optimization(
    current_params: Dict[str, float],
    objectives: Dict[str, bool],
    weights: Dict[str, float],
    num_iterations: int,
    population_size: int,
    mutation_rate: float,
    convergence_threshold: float,
    simulation_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Optimize tokenomics parameters based on specified objectives.
    
    Args:
        current_params: Current parameter values
        objectives: Dict of optimization objectives
        weights: Importance weights for each objective
        num_iterations: Number of optimization iterations
        population_size: Size of the population for genetic algorithm
        mutation_rate: Mutation rate for genetic algorithm
        convergence_threshold: Convergence threshold
        simulation_params: Additional simulation parameters
        
    Returns:
        Dict containing optimization results
    """
    # Define parameter bounds
    bounds = {
        'token_price': (0.1, 10.0),
        'burn_rate': (0.01, 0.20),
        'reward_decay_rate': (0.005, 0.05),
        'staking_apr': (0.02, 0.15),
        'transaction_fee_rate': (0.01, 0.10)
    }
    
    # Initialize optimization history
    history = []
    
    def objective_function(params):
        # Create simulation parameters by combining optimized params with fixed params
        sim_params = simulation_params.copy()
        sim_params.update({
            'initial_token_price': params[0],
            'burn_rate': params[1],
            'reward_decay_rate': params[2],
            'staking_apr': params[3],
            'transaction_fee_rate': params[4],
            'shock_events': None  # Disable shocks during optimization
        })
        
        # Simulate tokenomics with given parameters
        simulation_results = simulate_tokenomics(**sim_params)
        
        # Calculate objective scores
        scores = {}
        if objectives['token_price']:
            scores['token_price'] = calculate_token_price_score(simulation_results)
        if objectives['user_growth']:
            scores['user_growth'] = calculate_user_growth_score(simulation_results)
        if objectives['reserve_health']:
            scores['reserve_health'] = calculate_reserve_health_score(simulation_results)
        if objectives['token_velocity']:
            scores['token_velocity'] = calculate_token_velocity_score(simulation_results)
            
        # Calculate weighted sum
        weighted_score = sum(scores[obj] * weights[obj] for obj in objectives if objectives[obj])
        
        # Update history
        history.append({
            'params': dict(zip(bounds.keys(), params)),
            'scores': scores,
            'weighted_score': weighted_score
        })
        
        return -weighted_score  # Negative because we want to maximize
    
    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds=list(bounds.values()),
        maxiter=num_iterations,
        popsize=population_size,
        mutation=mutation_rate,
        tol=convergence_threshold
    )
    
    # Process results
    recommended_params = dict(zip(bounds.keys(), result.x))
    
    # Calculate convergence history
    convergence_history = pd.DataFrame([
        {
            'iteration': i,
            'best_score': -min(h['weighted_score'] for h in history[:i+1])
        }
        for i in range(len(history))
    ])
    
    return {
        'recommended_params': recommended_params,
        'convergence_history': convergence_history,
        'sensitivity_analysis': perform_sensitivity_analysis(history),
        'trade_offs': analyze_trade_offs(history)
    }
