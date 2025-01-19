import numpy as np
import pandas as pd
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def log_message(message, debug_only=False):
    if debug_only:
        logging.debug(message)
    else:
        logging.info(message)

def create_slider_with_range(
    label,
    default_min,
    default_max,
    default_value,
    step,
    format=None,
    key_prefix="",
    help_text=None,
):
    col_min, col_val, col_max = st.sidebar.columns([1, 2, 1])

    # Generate unique keys for each input
    min_key = f"{key_prefix}_min_{label}"
    max_key = f"{key_prefix}_max_{label}"
    slider_key = f"{key_prefix}_{label}"

    with col_min:
        min_value = st.number_input(
            f"Min", value=default_min, key=min_key, format=format
        )

    with col_max:
        max_value = st.number_input(
            f"Max", value=default_max, key=max_key, format=format
        )

    # Ensure min is less than max
    if min_value >= max_value:
        st.sidebar.error("Min value must be less than max value. Using default range.")
        min_value = default_min
        max_value = default_max

    # Initialize the slider with the current bounds
    with col_val:
        value = st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=min(max(default_value, min_value), max_value),
            step=step,
            format=format,
            key=slider_key,
            help=help_text,
        )

    return value

def simulate_tokenomics(
    initial_reward,
    initial_search_fee,
    growth_rate,
    line_items_per_customer,
    initial_lookup_frequency,
    reward_decay_rate,
    contribution_cap,
    initial_premium_adoption,
    inactivity_rate,
    months,
    base_users,
    customers_per_user,
    new_clients_per_user,
    initial_token_price,
    price_elasticity,
    burn_rate,
    initial_market_sentiment,
    market_volatility,
    market_trend,
    staking_apr,
    reward_pool_size,
    num_competitors,
    competitor_growth_rates,
    competitor_attractiveness,
    token_purchase_threshold,
    token_purchase_amount,
    token_sale_price,
    total_users_target,
    total_addressable_market,
    logistic_enabled=False,
    carrying_capacity=None,
    growth_steepness=0.3,
    midpoint_month=12,
    total_vested_tokens=0,
    vest_duration=0,
    shock_events=None
):
    """Simulate tokenomics with given parameters."""
    # Input validation and initialization
    if base_users < 0:
        raise ValueError("base_users cannot be negative")
    
    if total_addressable_market <= 0:
        raise ValueError("total_addressable_market must be greater than 0")
        
    if months <= 0:
        raise ValueError("months must be greater than 0")

    if initial_token_price <= 0:
        raise ValueError("initial_token_price must be positive")

    if reward_pool_size <= 0:
        raise ValueError("reward_pool_size must be positive")

    if initial_reward <= 0:
        raise ValueError("initial_reward must be positive")

    if initial_search_fee <= 0:
        raise ValueError("initial_search_fee must be positive")

    if contribution_cap <= 0:
        raise ValueError("contribution_cap must be positive")

    if new_clients_per_user < 0:
        raise ValueError("new_clients_per_user cannot be negative")

    # Validate shock events structure if provided
    if shock_events:
        if not isinstance(shock_events, list):
            raise ValueError("shock_events must be a list")
        
        for event in shock_events:
            if not isinstance(event, dict):
                raise ValueError("Each shock event must be a dictionary")
            
            if 'month' not in event:
                raise ValueError("Each shock event must have a 'month' field")
            
            if event['month'] < 1 or event['month'] > months:
                raise ValueError(f"Shock event month {event['month']} is outside simulation range")

    # Validate and adjust interdependent parameters
    if carrying_capacity is None:
        carrying_capacity = total_addressable_market

    if carrying_capacity < total_users_target:
        log_message("Warning: Carrying capacity is less than total users target. Adjusting target.", debug_only=True)
        total_users_target = carrying_capacity

    # Define initial seed value for zero-start case
    initial_seed = max(10, int(total_users_target * 0.001))  # Start with 0.1% of target or minimum 10 users

    # Ensure targets don't exceed TAM
    total_users_target = min(total_users_target, total_addressable_market)
    carrying_capacity = min(carrying_capacity, total_addressable_market)

    # Calculate maximum possible growth rate based on TAM with safety checks
    if base_users == 0:
        max_growth_rate = min(2.0, growth_rate)  # Cap initial growth at 200%
        log_message("Base users is 0, capping initial growth_rate at 200%", debug_only=True)
    else:
        try:
            max_growth_rate = (total_addressable_market / base_users) ** (1/months) - 1
            max_growth_rate = min(max_growth_rate, 2.0)  # Cap at 200% to prevent unrealistic growth
        except (ZeroDivisionError, ValueError):
            max_growth_rate = growth_rate
            log_message("Error calculating max_growth_rate, using input growth_rate", debug_only=True)

    effective_growth_rate = min(growth_rate, max_growth_rate)
    
    log_message(f"Base Users: {base_users}", debug_only=True)
    log_message(f"TAM: {total_addressable_market}", debug_only=True)
    log_message(f"Months: {months}", debug_only=True)
    log_message(f"Max Growth Rate: {max_growth_rate}", debug_only=True)
    log_message(f"Input Growth Rate: {growth_rate}", debug_only=True)
    log_message(f"Effective Growth Rate: {effective_growth_rate}", debug_only=True)

    # --- Initialization ---
    users = base_users
    total_tokens_earned = 0
    total_tokens_spent = 0
    total_tokens_burned = 0
    total_tokens_staked = 0
    reward_pool = reward_pool_size
    platform_revenue = 0
    reward = initial_reward
    search_fee = initial_search_fee
    premium_adoption = initial_premium_adoption
    lookup_frequency = initial_lookup_frequency
    token_price = initial_token_price
    market_sentiment = initial_market_sentiment
    monthly_token_spending = 0
    current_transaction_volume = 0
    previous_transaction_volume = 0
    monthly_spending = 0
    premium_spending = 0
    shock_descriptions = []
    shock_intensity = 1.0
    network_effect = 1.0
    market_maturity = 0.0
    seasonal_multiplier = 1.0

    # Initialize market phase based on initial market saturation
    initial_market_saturation = base_users / total_addressable_market
    market_phase = (
        "early" if initial_market_saturation < 0.1 else
        "growth" if initial_market_saturation < 0.5 else
        "mature"
    )

    # Initialize tracking variables
    previous_price = initial_token_price
    price_history = []
    sentiment_history = []
    price_change_history = []

    # Initialize segment counts
    segment_counts = {
        segment: int(users * data["proportion"]) 
        for segment, data in user_segments.items()
    }

    # Log initial state
    log_message(f"Initial market phase: {market_phase}", debug_only=True)
    log_message(f"Initial market saturation: {initial_market_saturation:.2%}", debug_only=True)

    # Update rewards with decay
    reward = reward * (1 - reward_decay_rate)

    # Initialize financial metrics
    monthly_spending = 0
    premium_spending = 0
    monthly_token_spending = 0

    # Calculate competitor impact
    competitor_impact = 1.0
    for i in range(num_competitors):
        competitor_growth = competitor_growth_rates[i] if isinstance(competitor_growth_rates, list) else competitor_growth_rates
        competitor_attract = competitor_attractiveness[i] if isinstance(competitor_attractiveness, list) else competitor_attractiveness
        
        # Competitors have varying impact based on market phase
        phase_impact = {
            "early": 0.7,    # Less impact in early market
            "growth": 1.0,   # Full impact during growth
            "mature": 0.8    # Reduced impact in mature market
        }[market_phase]
        
        competitor_strength = (competitor_attract * (1 + competitor_growth)) * phase_impact
        competitor_impact *= (1 - competitor_strength * (1 - market_saturation))

    # Calculate staking behavior
    staking_propensity = {
        "power": 0.3,     # 30% likely to stake
        "regular": 0.15,  # 15% likely to stake
        "casual": 0.05    # 5% likely to stake
    }

    # Calculate target staking amount based on market conditions
    target_staking = sum(
        segment_counts[segment] * staking_propensity[segment] * total_tokens_earned
        for segment in segment_counts
    ) * (1 + (market_sentiment - 1) * 0.5)  # Sentiment boosts staking

    # Adjust staking based on price movement
    if token_price > previous_price:
        # More staking when price rises
        stake_adjustment = (target_staking - total_tokens_staked) * 0.2  # 20% adjustment toward target
    else:
        # Some unstaking when price falls
        stake_adjustment = -total_tokens_staked * 0.05  # 5% unstaking

    # Apply staking changes with limits
    max_stake_change = total_tokens_earned * 0.2  # Max 20% change per month
    stake_adjustment = np.clip(stake_adjustment, -max_stake_change, max_stake_change)
    total_tokens_staked = max(0, total_tokens_staked + stake_adjustment)

    # Calculate staking rewards
    staking_rewards = total_tokens_staked * (staking_apr / 12)  # Monthly rewards
    if reward_pool < staking_rewards:
        staking_rewards = reward_pool  # Cap rewards at available pool
        log_message("Warning: Reward pool insufficient for full staking rewards", debug_only=True)

    # Update reward pool
    reward_pool -= staking_rewards

    # Update reward with decay and pool status
    if reward_pool > 0:
        reward = reward * (1 - reward_decay_rate)
    else:
        reward = 0
        log_message("Warning: Reward pool depleted", debug_only=True)

    # Behavioral Economics Components
    def calculate_behavioral_metrics(
        market_phase,
        market_saturation,
        price_trend,
        token_price,
        initial_token_price,
        market_sentiment,
        network_validation,
        platform_revenue,
        users,
        price_change_history,
        total_tokens_staked,
        total_tokens_earned,
        competitor_impact
    ):
        """Calculate behavioral metrics with improved psychology factors"""
        
        # 1. Risk Perception
        volatility = np.std(price_change_history[-10:]) if len(price_change_history) >= 10 else 0
        phase_risk = {
            "early": 0.7,    # High risk perception
            "growth": 0.4,   # Moderate risk
            "mature": 0.2    # Lower risk
        }[market_phase]
        
        # Risk increases with volatility but plateaus
        perceived_risk = phase_risk * (1 + np.tanh(volatility * 2))
        
        # 2. Loss Aversion
        # Losses hurt more than equivalent gains (prospect theory)
        loss_aversion_base = 2.5  # Standard prospect theory value
        recent_losses = [x for x in price_change_history[-5:] if x < 0]
        loss_frequency = len(recent_losses) / 5 if len(price_change_history) >= 5 else 0
        
        loss_aversion_multiplier = loss_aversion_base * (1 + loss_frequency)
        recent_performance = price_trend * (loss_aversion_multiplier if price_trend < 0 else 1.0)
        
        # 3. Anchoring Effects
        # Price anchoring with diminishing effect over time
        anchor_ratio = token_price / initial_token_price
        historical_anchor = np.mean(price_change_history[-6:]) if len(price_change_history) >= 6 else 0
        time_decay = np.exp(-0.1 * len(price_change_history))  # Diminishing anchor effect
        anchoring_effect = np.exp(-0.5 * (anchor_ratio - 1 - historical_anchor)**2) * (1 + time_decay)
        
        # 4. Social Proof and Network Effects
        # Adoption thresholds with network validation
        adoption_threshold = {
            "early": 0.1,    # Early adopters need less proof
            "growth": 0.3,   # Growth phase needs moderate proof
            "mature": 0.5    # Mature phase needs strong proof
        }[market_phase]
        
        # Network validation increases with market maturity
        social_validation = min(1.0, network_validation / adoption_threshold)
        network_effect = np.log1p(market_saturation) * (1 + social_validation)
        
        # 5. Herding Behavior
        # Stronger herding in growth phase
        phase_herd_factors = {
            "early": 0.2,    # Less herding in early phase
            "growth": 0.4,   # Strong herding in growth
            "mature": 0.3    # Moderate herding in mature
        }[market_phase]
        
        sentiment_momentum = np.mean(price_change_history[-3:]) if len(price_change_history) >= 3 else 0
        herd_factor = phase_herd_factors * sentiment_momentum * (1 - market_saturation)
        
        # 6. Utility Value Perception
        # Value perception based on actual utility and network effects
        utility_per_user = platform_revenue / users if users > 0 else 0
        perceived_utility = utility_per_user * (1 + network_effect)
        
        # 7. Staking Psychology
        # Staking propensity affected by market conditions
        stake_ratio = total_tokens_staked / total_tokens_earned if total_tokens_earned > 0 else 0
        optimal_stake_ratio = {
            "early": 0.3,    # Lower optimal stake in early phase
            "growth": 0.4,   # Moderate optimal stake in growth
            "mature": 0.5    # Higher optimal stake in mature
        }[market_phase]
        
        staking_confidence = 1 - abs(stake_ratio - optimal_stake_ratio)
        
        # 8. Competition Psychology
        # Impact of competition on user psychology
        competition_anxiety = competitor_impact * (1 - market_saturation)
        
        # Calculate final behavioral sentiment
        behavioral_components = {
            'risk': (1 - perceived_risk) * 0.15,           # 15% weight
            'performance': recent_performance * 0.15,       # 15% weight
            'anchoring': anchoring_effect * 0.15,          # 15% weight
            'social': social_validation * 0.15,            # 15% weight
            'herding': herd_factor * 0.10,                 # 10% weight
            'utility': (perceived_utility > 0) * 0.15,     # 15% weight
            'staking': staking_confidence * 0.10,          # 10% weight
            'competition': (1 - competition_anxiety) * 0.05 # 5% weight
        }
        
        behavioral_sentiment = sum(behavioral_components.values())
        
        # Apply phase-specific bounds
        sentiment_bounds = {
            "early": (0.2, 3.0),    # Wide range in early phase
            "growth": (0.4, 2.0),   # Moderate range in growth
            "mature": (0.6, 1.5)    # Narrow range in mature phase
        }[market_phase]
        
        behavioral_sentiment = np.clip(behavioral_sentiment, *sentiment_bounds)
        
        return {
            'behavioral_sentiment': behavioral_sentiment,
            'components': behavioral_components,
            'metrics': {
                'perceived_risk': perceived_risk,
                'loss_aversion': loss_aversion_multiplier,
                'network_effect': network_effect,
                'social_validation': social_validation,
                'anchoring_effect': anchoring_effect,
                'perceived_utility': perceived_utility,
                'staking_confidence': staking_confidence,
                'competition_anxiety': competition_anxiety
            }
        }

    # Economic Incentive Mechanisms
    def calculate_economic_metrics(
        users,
        platform_revenue,
        reward,
        reward_pool,
        total_tokens_staked,
        total_tokens_earned,
        monthly_token_spending,
        circulating_supply,
        market_phase,
        market_maturity
    ):
        # 1. Dynamic Reward Adjustment
        utility_per_user = platform_revenue / users if users > 0 else 0
        target_reward_ratio = {
            "early": 0.4,    # Higher rewards in early phase
            "growth": 0.3,   # Moderate rewards in growth
            "mature": 0.2    # Lower rewards in mature phase
        }[market_phase]
        
        current_reward_ratio = reward / utility_per_user if utility_per_user > 0 else 1
        reward_adjustment = (target_reward_ratio - current_reward_ratio) * 0.2
        
        # Pool sustainability check
        reward_sustainability = reward_pool / (users * reward * 12) if users * reward > 0 else 0
        sustainability_factor = np.clip(reward_sustainability / 12, 0, 1)  # Target 1 year runway
        
        # Adjust reward based on sustainability
        if reward_sustainability < 6:
            reward_adjustment *= 0.5  # Reduce adjustment if low runway
        elif reward_sustainability > 24:
            reward_adjustment *= 1.2  # Increase adjustment if high runway
        
        new_reward = reward * (1 + reward_adjustment * sustainability_factor)
        reward = np.clip(new_reward, reward * 0.5, reward * 1.5)
        
        # 2. Dynamic Fee Structure
        base_discounts = {
            "early": {
                "volume": 0.1,     # 10% volume discount
                "loyalty": 0.1     # 10% loyalty discount
            },
            "growth": {
                "volume": 0.2,     # 20% volume discount
                "loyalty": 0.15    # 15% loyalty discount
            },
            "mature": {
                "volume": 0.3,     # 30% volume discount
                "loyalty": 0.2     # 20% loyalty discount
            }
        }[market_phase]
        
        # Calculate effective discounts
        volume_discount = base_discounts["volume"] * (1 + market_maturity)
        loyalty_discount = min(base_discounts["loyalty"], market_maturity * 0.3)
        network_premium = np.log1p(market_maturity) * 0.1  # Up to 10% premium for network value
        
        # 3. Staking Economics
        # Calculate optimal staking ratio based on market phase
        optimal_stake_ratios = {
            "early": 0.3,    # Lower optimal stake in early phase
            "growth": 0.4,   # Moderate optimal stake in growth
            "mature": 0.5    # Higher optimal stake in mature phase
        }
        
        base_optimal_ratio = optimal_stake_ratios[market_phase]
        optimal_stake_ratio = base_optimal_ratio + (market_maturity * 0.2)  # Increases with maturity
        current_stake_ratio = total_tokens_staked / total_tokens_earned if total_tokens_earned > 0 else 0
        
        # Dynamic APR adjustment
        stake_ratio_gap = optimal_stake_ratio - current_stake_ratio
        apr_adjustment = np.clip(stake_ratio_gap * 2, -0.3, 0.3)  # ±30% max adjustment
        
        # 4. Token Utility Metrics
        token_velocity = (monthly_token_spending * 12) / circulating_supply if circulating_supply > 0 else 0
        target_velocity = {
            "early": 15,     # Higher velocity in early phase
            "growth": 12,    # Moderate velocity in growth
            "mature": 8      # Lower velocity in mature phase
        }[market_phase]
        
        velocity_alignment = 1 - abs(token_velocity - target_velocity) / target_velocity
        
        # 5. Economic Stability Score
        stability_components = {
            "velocity": velocity_alignment,
            "staking": 1 - abs(current_stake_ratio - optimal_stake_ratio),
            "sustainability": sustainability_factor
        }
        
        # Weight components based on market phase
        stability_weights = {
            "early": {
                "velocity": 0.5,           # Focus on activity
                "staking": 0.2,            # Less focus on staking
                "sustainability": 0.3       # Moderate focus on sustainability
            },
            "growth": {
                "velocity": 0.3,           # Balanced velocity importance
                "staking": 0.4,            # Increased staking importance
                "sustainability": 0.3       # Maintained sustainability focus
            },
            "mature": {
                "velocity": 0.2,           # Reduced velocity importance
                "staking": 0.5,            # High staking importance
                "sustainability": 0.3       # Consistent sustainability focus
            }
        }[market_phase]
        
        stability_score = sum(
            stability_weights[metric] * value
            for metric, value in stability_components.items()
        )
        
        return {
            'reward': reward,
            'reward_sustainability': reward_sustainability,
            'volume_discount': volume_discount,
            'loyalty_discount': loyalty_discount,
            'network_premium': network_premium,
            'optimal_stake_ratio': optimal_stake_ratio,
            'apr_adjustment': apr_adjustment,
            'token_velocity': token_velocity,
            'stability_score': stability_score
        }

    # Calculate token burn and supply dynamics
    def calculate_burn_dynamics(
        monthly_token_spending,
        burn_rate,
        total_tokens_burned,
        total_tokens_earned,
        total_tokens_staked,
        market_phase,
        market_maturity,
        token_velocity,
        platform_revenue,
        users
    ):
        """Calculate token burn dynamics with economic constraints"""
        
        # Base burn calculation
        base_burn = monthly_token_spending * burn_rate
        
        # Calculate burn/supply ratio
        circulating_supply = max(1, total_tokens_earned - total_tokens_burned - total_tokens_staked)
        burn_supply_ratio = total_tokens_burned / total_tokens_earned if total_tokens_earned > 0 else 0
        
        # Target burn ratios by market phase with economic justification
        target_burn_ratios = {
            "early": 0.4,    # Higher burn to establish scarcity
            "growth": 0.3,   # Moderate burn to balance growth
            "mature": 0.2    # Lower burn to maintain stability
        }[market_phase]
        
        # Calculate utility per token
        utility_per_token = (platform_revenue / users) / token_velocity if users > 0 and token_velocity > 0 else 0
        
        # Adjust target ratio based on utility
        utility_factor = np.clip(utility_per_token / base_burn, 0.5, 1.5)
        adjusted_target = target_burn_ratios * utility_factor
        
        # Calculate burn pressure
        burn_gap = adjusted_target - burn_supply_ratio
        burn_pressure = np.clip(burn_gap * 2, -0.5, 0.5)
        
        # Dynamic burn rate adjustment
        phase_modifiers = {
            "early": 1.2,    # Aggressive burn in early phase
            "growth": 1.0,   # Standard burn in growth
            "mature": 0.8    # Conservative burn in mature
        }[market_phase]
        
        # Supply-based adjustment
        min_circulating = total_tokens_earned * 0.1  # Maintain 10% minimum circulating
        if circulating_supply < min_circulating:
            supply_factor = 0.5  # Reduce burn rate
            log_message("Warning: Low circulating supply, reducing burn rate", debug_only=True)
        elif circulating_supply > total_tokens_earned * 0.8:
            supply_factor = 1.2  # Increase burn rate
            log_message("Notice: High circulating supply, increasing burn rate", debug_only=True)
        else:
            supply_factor = 1.0
        
        # Calculate effective burn multiplier
        burn_multiplier = phase_modifiers * supply_factor * (1 + burn_pressure)
        
        # Calculate effective burn amount with constraints
        effective_burn = base_burn * burn_multiplier
        
        # Apply safety bounds
        max_burn = monthly_token_spending * 0.5  # Never burn more than 50% of spending
        min_burn = monthly_token_spending * 0.01  # Always burn at least 1% of spending
        effective_burn = np.clip(effective_burn, min_burn, max_burn)
        
        # Calculate supply impact
        supply_impact = burn_pressure * (1 - market_maturity)
        
        # Ensure minimum circulating supply
        if (circulating_supply - effective_burn) < min_circulating:
            effective_burn = max(0, circulating_supply - min_circulating)
            log_message("Warning: Burn amount adjusted to maintain minimum circulating supply", debug_only=True)
        
        return {
            'burn_amount': effective_burn,
            'burn_pressure': burn_pressure,
            'supply_impact': supply_impact,
            'burn_supply_ratio': burn_supply_ratio,
            'adjusted_burn_rate': burn_rate * burn_multiplier,
            'utility_factor': utility_factor,
            'supply_factor': supply_factor
        }

    # Market Phase and Price Formation
    def calculate_market_phase(market_saturation, current_phase):
        """Calculate market phase with hysteresis to prevent rapid switching"""
        thresholds = {
            "early": {"lower": 0.0, "upper": 0.1},    # 0-10%
            "growth": {"lower": 0.1, "upper": 0.5},   # 10-50%
            "mature": {"lower": 0.5, "upper": 1.0}    # 50-100%
        }
        
        # Add hysteresis buffer (10% of threshold)
        if current_phase == "mature":
            if market_saturation < thresholds["growth"]["upper"] * 0.9:
                return "growth"
        elif current_phase == "growth":
            if market_saturation < thresholds["early"]["upper"] * 0.9:
                return "early"
            elif market_saturation > thresholds["mature"]["lower"]:
                return "mature"
        elif current_phase == "early":
            if market_saturation > thresholds["growth"]["lower"]:
                return "growth"
        
        return current_phase

    def calculate_price_components(
        token_price,
        total_demand,
        effective_supply,
        market_depth,
        market_sentiment,
        market_maturity,
        market_phase,
        fundamental_price,
        price_history,
        network_effect,
        competitor_impact
    ):
        """Calculate price change components with proper weights and constraints"""
        
        # 1. Supply-Demand Impact (35% weight)
        supply_demand_ratio = total_demand / effective_supply if effective_supply > 0 else 1
        supply_demand_impact = np.clip((supply_demand_ratio - 1) * 0.35, -0.35, 0.35)
        
        # 2. Market Depth Impact (15% weight)
        # Deeper markets have less price impact
        depth_ratio = market_depth / (total_demand * 10) if total_demand > 0 else 1
        depth_impact = np.clip((1 - depth_ratio) * 0.15, -0.15, 0.15)
        
        # 3. Sentiment Impact (20% weight)
        # Sentiment impact decreases with market maturity
        raw_sentiment = (market_sentiment - 1) * (1 - market_maturity)
        sentiment_impact = np.clip(raw_sentiment * 0.2, -0.2, 0.2)
        
        # 4. Network Effect (15% weight)
        # Network effect strengthens with maturity
        network_impact = np.clip(network_effect * market_maturity * 0.15, 0, 0.15)
        
        # 5. Fundamental Value Impact (15% weight)
        # Stronger pull towards fundamental value in mature markets
        value_gap = (fundamental_price / token_price - 1) if token_price > 0 else 0
        fundamental_impact = np.clip(
            value_gap * (0.15 + market_maturity * 0.1),  # Increases with maturity
            -0.15,
            0.15
        )
        
        # Calculate base price change
        price_change = (
            supply_demand_impact +
            depth_impact +
            sentiment_impact +
            network_impact +
            fundamental_impact
        )
        
        # Apply phase-specific volatility constraints
        volatility_bounds = {
            "early": {"down": -0.4, "up": 0.5},     # Higher volatility
            "growth": {"down": -0.3, "up": 0.4},    # Moderate volatility
            "mature": {"down": -0.2, "up": 0.3}     # Lower volatility
        }[market_phase]
        
        # Adjust bounds based on market depth
        depth_factor = np.clip(depth_ratio, 0.5, 1.5)
        final_bounds = (
            volatility_bounds["down"] * depth_factor,
            volatility_bounds["up"] * depth_factor
        )
        
        # Apply competitor impact
        competitor_dampening = 1 - (competitor_impact * 0.2)  # Up to 20% reduction
        price_change *= competitor_dampening
        
        # Final bounds check
        price_change = np.clip(price_change, *final_bounds)
        
        return {
            'price_change': price_change,
            'components': {
                'supply_demand': supply_demand_impact,
                'depth': depth_impact,
                'sentiment': sentiment_impact,
                'network': network_impact,
                'fundamental': fundamental_impact,
                'competitor': competitor_dampening
            }
        }

    # --- Simulation Loop ---
    for month in range(months):
        current_month = month + 1
        
        # Special handling for first month if starting from 0
        if month == 0:
            if users == 0:
                users = initial_seed
                log_message(f"Month 1: Seeding initial user base with {initial_seed} user(s)", debug_only=True)
            
            # Initialize financial metrics
            monthly_spending = 0
            premium_spending = 0
            monthly_token_spending = 0
            current_transaction_volume = 0
            price_trend = 0
            network_validation = 0
            
            # Initialize segment counts
            segment_counts = {
                segment: int(users * data["proportion"]) 
                for segment, data in user_segments.items()
            }
        
        # Calculate market metrics
        market_saturation = min(users / total_addressable_market, 1.0) if total_addressable_market > 0 else 1.0
        market_maturity = np.sqrt(market_saturation)  # Non-linear maturity scaling
        
        # Update market phase with hysteresis
        market_phase = calculate_market_phase(market_saturation, market_phase)
        
        # Calculate effective growth with safety bounds
        if users == 0:
            effective_growth = min(2.0, growth_rate)  # Cap initial growth at 200%
        else:
            months_remaining = max(1, months - current_month + 1)
            required_growth = ((total_users_target / users) ** (1/months_remaining) - 1) if total_users_target > users else 0
            base_growth = min(growth_rate, required_growth)
            
            # Apply market phase and saturation effects
            phase_growth_modifiers = {
                "early": 1.2,    # Higher growth in early phase
                "growth": 1.0,   # Normal growth in growth phase
                "mature": 0.8    # Reduced growth in mature phase
            }
            
            saturation_factor = max(0, 1 - (market_saturation ** 1.5))  # Steeper decline near saturation
            effective_growth = base_growth * phase_growth_modifiers[market_phase] * saturation_factor
        
        # Calculate new users with bounds
        new_users = int(max(0, users * effective_growth))
        if users == 0:
            new_users = initial_seed
        
        # Calculate churn with improved dynamics
        base_churn = inactivity_rate * (1 + market_saturation * 0.5)  # Reduced saturation impact
        phase_churn_modifiers = {
            "early": 1.2,    # Higher churn in early phase
            "growth": 1.0,   # Normal churn in growth phase
            "mature": 0.8    # Lower churn in mature phase
        }
        
        # Apply segment-specific churn with safety checks
        churned_users = 0
        for segment, count in segment_counts.items():
            segment_churn = (
                count * 
                base_churn * 
                phase_churn_modifiers[market_phase] / 
                max(0.1, user_segments[segment]["churn_resistance"])  # Prevent division by zero
            )
            churned_users += int(max(0, segment_churn))
        
        # Update user count with improved bounds
        users += new_users - churned_users
        users = min(users, int(carrying_capacity * 0.95))  # Never exceed 95% of carrying capacity
        users = max(users, int(base_users * 0.5))  # Never drop below 50% of base users
        
        # Update segment counts with proper rounding
        total_proportion = sum(data["proportion"] for data in user_segments.values())
        remaining_users = users
        for segment in list(segment_counts.keys())[:-1]:  # Process all but last segment
            count = int(users * user_segments[segment]["proportion"] / total_proportion)
            segment_counts[segment] = count
            remaining_users -= count
        # Assign remaining users to last segment to ensure total adds up
        segment_counts[list(segment_counts.keys())[-1]] = remaining_users
        
        # Calculate token economics with safety checks
        circulating_supply = max(1, total_tokens_earned - total_tokens_burned - total_tokens_staked)
        effective_supply = circulating_supply * (1 + supply_impact if 'supply_impact' in locals() else 1.0)
        
        # Calculate demand components with safety
        base_demand = monthly_token_spending if 'monthly_token_spending' in locals() else 0
        speculative_demand = token_purchase_amount if token_price <= token_purchase_threshold else 0
        utility_demand = base_demand * max(0, 1 + network_effect - 1)  # Ensure non-negative
        total_demand = max(0, base_demand + speculative_demand + utility_demand)
        
        # Calculate market depth with minimum liquidity
        market_depth = max(
            circulating_supply * token_price * market_maturity,
            total_demand * 0.1  # Ensure minimum liquidity
        )
        
        # Calculate price components and update token price
        price_metrics = calculate_price_components(
            token_price=token_price,
            total_demand=total_demand,
            effective_supply=effective_supply,
            market_depth=market_depth,
            market_sentiment=market_sentiment,
            market_maturity=market_maturity,
            market_phase=market_phase,
            fundamental_price=fundamental_price if 'fundamental_price' in locals() else initial_token_price,
            price_history=price_history,
            network_effect=network_effect,
            competitor_impact=competitor_impact
        )
        
        price_change = price_metrics['price_change']
        components = price_metrics['components']
        
        # Update token price with safety bounds
        previous_price = token_price
        token_price = token_price * (1 + price_change)
        token_price = max(token_price, initial_token_price * 0.1)  # Prevent near-zero prices
        
        # Calculate price trend for sentiment
        price_trend = (token_price - previous_price) / previous_price if previous_price > 0 else 0
        
        # Update behavioral metrics
        behavioral_metrics = calculate_behavioral_metrics(
            market_phase=market_phase,
            market_saturation=market_saturation,
            price_trend=price_trend,
            token_price=token_price,
            initial_token_price=initial_token_price,
            market_sentiment=market_sentiment,
            network_validation=network_validation,
            platform_revenue=platform_revenue if 'platform_revenue' in locals() else 0,
            users=users,
            price_change_history=price_change_history,
            total_tokens_staked=total_tokens_staked,
            total_tokens_earned=total_tokens_earned,
            competitor_impact=competitor_impact
        )
        
        # Update market sentiment and network effect
        market_sentiment = behavioral_metrics['behavioral_sentiment']
        network_effect = behavioral_metrics['metrics']['network_effect']
        
        # Update burn dynamics
        burn_metrics = calculate_burn_dynamics(
            monthly_token_spending=monthly_token_spending if 'monthly_token_spending' in locals() else 0,
            burn_rate=burn_rate,
            total_tokens_burned=total_tokens_burned,
            total_tokens_earned=total_tokens_earned,
            total_tokens_staked=total_tokens_staked,
            market_phase=market_phase,
            market_maturity=market_maturity,
            token_velocity=token_velocity if 'token_velocity' in locals() else 0,
            platform_revenue=platform_revenue if 'platform_revenue' in locals() else 0,
            users=users
        )
        
        # Update burn-related variables
        burned_tokens = burn_metrics['burn_amount']
        burn_pressure = burn_metrics['burn_pressure']
        supply_impact = burn_metrics['supply_impact']
        burn_rate = burn_metrics['adjusted_burn_rate']
        
        # Update total burned tokens with safety check
        total_tokens_burned = min(
            total_tokens_burned + burned_tokens,
            total_tokens_earned * 0.9  # Never burn more than 90% of total supply
        )
        
        # Update tracking variables
        price_history.append(token_price)
        price_change_history.append(price_change)
        sentiment_history.append(market_sentiment)
        
        # Handle edge cases
        if total_tokens_earned == 0:
            token_price = initial_token_price
            market_sentiment = initial_market_sentiment
        
        # Prevent negative reward pool
        reward_pool = max(0, reward_pool)
        if reward_pool == 0:
            log_message("Warning: Reward pool depleted", debug_only=True)
            reward = 0  # Stop rewards when pool is depleted
        
        # Apply shock events if any
        if shock_events and isinstance(shock_events, list):
            for event in shock_events:
                if event['month'] == current_month:
                    apply_shock_event(event, locals())  # Apply shock with proper variable scope
        
        # Collect monthly results with proper initialization checks
        monthly_results.append(collect_monthly_metrics(locals()))

    return pd.DataFrame(monthly_results)

def get_simulation_inputs():
    """Get all simulation inputs from the Streamlit UI."""
    initial_reward = st.sidebar.number_input("Initial Reward", value=100.0)
    initial_search_fee = st.sidebar.number_input("Initial Search Fee", value=0.1)
    growth_rate = st.sidebar.slider("Growth Rate", min_value=0.0, max_value=1.0, value=0.05)
    line_items_per_customer = st.sidebar.number_input("Line Items per Customer", value=10)
    initial_lookup_frequency = st.sidebar.number_input("Initial Lookup Frequency", value=5)
    reward_decay_rate = st.sidebar.slider("Reward Decay Rate", min_value=0.0, max_value=1.0, value=0.01)
    contribution_cap = st.sidebar.number_input("Monthly Token Contribution Cap", value=1000)
    initial_premium_adoption = st.sidebar.slider("Initial Premium Adoption", min_value=0.0, max_value=1.0, value=0.1)
    inactivity_rate = st.sidebar.slider("Inactivity Rate", min_value=0.0, max_value=1.0, value=0.05)
    months = st.sidebar.number_input("Months", value=12)
    base_users = st.sidebar.number_input("Base Users", value=1000)
    customers_per_user = st.sidebar.number_input("Customers per User", value=5)
    new_clients_per_user = st.sidebar.number_input("New Clients per User", value=1)
    initial_token_price = st.sidebar.number_input("Initial Token Price", value=1.0)
    price_elasticity = st.sidebar.slider("Price Elasticity", min_value=0.0, max_value=2.0, value=1.0)
    burn_rate = st.sidebar.slider("Burn Rate", min_value=0.0, max_value=1.0, value=0.01)
    initial_market_sentiment = st.sidebar.slider("Initial Market Sentiment", min_value=0.0, max_value=2.0, value=1.0)
    market_volatility = st.sidebar.slider("Market Volatility", min_value=0.0, max_value=1.0, value=0.1)
    market_trend = st.sidebar.slider("Market Trend", min_value=-1.0, max_value=1.0, value=0.0)
    staking_apr = st.sidebar.slider("Staking APR", min_value=0.0, max_value=0.5, value=0.1)
    reward_pool_size = st.sidebar.number_input("Reward Pool Size", value=10000)
    num_competitors = st.sidebar.number_input("Number of Competitors", value=5)
    competitor_growth_rates = st.sidebar.slider("Competitor Growth Rates", min_value=0.0, max_value=1.0, value=0.05)
    competitor_attractiveness = st.sidebar.slider("Competitor Attractiveness", min_value=0.0, max_value=1.0, value=0.5)
    token_purchase_threshold = st.sidebar.number_input("Token Purchase Threshold", value=10)
    token_purchase_amount = st.sidebar.number_input("Token Purchase Amount", value=100)
    token_sale_price = st.sidebar.number_input("Token Sale Price", value=1.0)
    total_users_target = st.sidebar.number_input("Total Users Target", value=10000)
    total_addressable_market = st.sidebar.number_input("Total Addressable Market", value=100000)
    logistic_enabled = st.sidebar.checkbox("Logistic Enabled", value=False)
    carrying_capacity = st.sidebar.number_input("Carrying Capacity", value=100000)
    growth_steepness = st.sidebar.slider("Growth Steepness", min_value=0.0, max_value=1.0, value=0.3)
    midpoint_month = st.sidebar.number_input("Midpoint Month", value=12)
    total_vested_tokens = st.sidebar.number_input("Total Vested Tokens", value=0)
    vest_duration = st.sidebar.number_input("Vest Duration", value=0)
    shock_events = None  # Assuming no UI for shock events for simplicity

    return (
        initial_reward, initial_search_fee, growth_rate, line_items_per_customer,
        initial_lookup_frequency, reward_decay_rate, contribution_cap,
        initial_premium_adoption, inactivity_rate, months, base_users,
        customers_per_user, new_clients_per_user, initial_token_price,
        price_elasticity, burn_rate, initial_market_sentiment, market_volatility,
        market_trend, staking_apr, reward_pool_size, num_competitors,
        competitor_growth_rates, competitor_attractiveness, None,
        token_purchase_threshold, token_purchase_amount, token_sale_price,
        total_users_target, total_addressable_market, logistic_enabled,
        carrying_capacity, growth_steepness, midpoint_month, total_vested_tokens,
        vest_duration, shock_events
    )

def main():
    # Get simulation inputs from UI
    simulation_params = get_simulation_inputs()
    
    # Run simulation with the inputs
    results = simulate_tokenomics(*simulation_params)
    
    return results

if __name__ == "__main__":
    results = main()
