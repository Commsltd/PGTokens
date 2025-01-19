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

def calculate_token_emission(
    current_supply,
    target_supply,
    emission_rate,
    market_maturity,
    network_growth,
    max_monthly_change=0.1
):
    """Calculate controlled token emission with real-world constraints."""
    supply_gap = target_supply - current_supply
    base_emission = supply_gap * emission_rate
    
    adjusted_emission = base_emission * (
        (1 - market_maturity * 0.5) *     # Lower emissions in mature markets
        (1 + network_growth * 0.3)        # Slightly higher during growth
    )
    
    max_emission = current_supply * max_monthly_change
    return np.clip(adjusted_emission, -max_emission, max_emission)

def manage_treasury_tokens(
    treasury_balance,
    circulating_supply,
    market_conditions,
    min_treasury_ratio=0.2,
    max_release_rate=0.05
):
    """Manage treasury tokens with institutional-grade controls."""
    target_treasury = circulating_supply * min_treasury_ratio
    max_release = treasury_balance * max_release_rate
    
    if market_conditions['demand'] > market_conditions['supply']:
        release_amount = max_release * market_conditions['demand_pressure']
    else:
        release_amount = 0
    
    if (treasury_balance - release_amount) < target_treasury:
        release_amount = max(0, treasury_balance - target_treasury)
        
    return release_amount

def manage_treasury_changes(
    current_treasury,
    reward_pool_size,
    revenue_change,
    emission_change,
    market_maturity,
    previous_changes=None,
    token_velocity=None,
    market_phase=None
):
    """Manage treasury changes with enhanced stability controls"""
    if previous_changes is None:
        previous_changes = []
    
    # Calculate base change with initial dampening
    total_change = (revenue_change + emission_change) * 0.8  # Initial 20% reduction
    
    # Progressive dampening based on change size
    change_ratio = abs(total_change / current_treasury) if current_treasury > 0 else float('inf')
    dampening_factors = [
        (0.01, 1.0),      # Up to 1%: no dampening
        (0.03, 0.7),      # 1-3%: 30% dampening
        (0.05, 0.5),      # 3-5%: 50% dampening
        (0.08, 0.3),      # 5-8%: 70% dampening
        (0.10, 0.2),      # 8-10%: 80% dampening
        (float('inf'), 0.1)  # Above 10%: 90% dampening
    ]
    
    # Apply stricter phase-based dampening
    phase_multipliers = {
        "early": 0.8,     # 20% additional dampening in early phase
        "growth": 0.7,    # 30% additional dampening in growth phase
        "mature": 0.6     # 40% additional dampening in mature phase
    }
    phase_mult = phase_multipliers.get(market_phase, 0.7)
    
    # Find and apply appropriate dampening
    for threshold, factor in dampening_factors:
        if change_ratio <= threshold:
            total_change *= factor * phase_mult
            break
    
    # Enhanced momentum check
    if len(previous_changes) >= 3:
        recent_growth = np.mean(previous_changes[-3:]) / current_treasury
        volatility = np.std(previous_changes[-3:]) / current_treasury
        
        # Stronger dampening for high growth or volatility
        if abs(recent_growth) > 0.03:  # Reduced from 0.05
            total_change *= 0.5   # Increased dampening
        if volatility > 0.02:     # Reduced from 0.03
            total_change *= 0.6   # Increased dampening
    
    # Enhanced market maturity impact
    maturity_cap = (1 - market_maturity * 0.8)  # Increased from 0.7
    max_monthly_change = current_treasury * 0.03 * maturity_cap  # Reduced from 0.05
    
    # Apply velocity-based adjustment if available
    if token_velocity is not None:
        velocity_factor = np.clip(1 / (1 + token_velocity * 0.2), 0.3, 1.0)  # Stronger velocity impact
        max_monthly_change *= velocity_factor
    
    # Apply monthly change limit
    total_change = np.clip(total_change, -max_monthly_change, max_monthly_change)
    
    # Enhanced pool size constraints
    max_treasury = reward_pool_size * (
        0.9 * (1 - market_maturity * 0.5)  # Reduced multiplier
    )
    min_treasury = reward_pool_size * 0.3   # Increased minimum
    
    # Calculate final treasury with enhanced bounds
    new_treasury = np.clip(
        current_treasury + total_change,
        min_treasury,
        max_treasury
    )
    
    # Final circuit breaker
    max_change_ratio = 0.05  # Reduced from 0.1
    actual_change = np.clip(
        new_treasury - current_treasury,
        -current_treasury * max_change_ratio,
        current_treasury * max_change_ratio
    )
    
    new_treasury = current_treasury + actual_change
    
    return actual_change, new_treasury

def calculate_adaptive_reward(
    base_reward,
    engagement_score,
    value_per_user,
    network_effect,
    market_maturity,
    reward_pool_ratio,
    price_ratio,
    supply_ratio,
    token_velocity,
    inflation_target=0.05  # 5% annual inflation target
):
    """Calculate adaptive reward with anti-inflation controls"""
    # Calculate current inflation rate
    inflation_rate = max(0, (price_ratio - 1) * token_velocity)
    inflation_gap = inflation_rate - inflation_target
    
    # Anti-inflation adjustment
    inflation_dampener = 1 / (1 + max(0, inflation_gap) * 2)  # Stronger dampening for high inflation
    
    # Calculate base adjustment factors
    utility_factor = np.clip(value_per_user / base_reward, 0.5, 2.0)
    network_factor = np.clip(1 + network_effect * 0.3, 0.8, 1.5)
    maturity_factor = 1 - market_maturity * 0.4  # Reduce rewards in mature markets
    
    # Supply-based adjustment
    supply_adjustment = np.clip(1 / supply_ratio, 0.5, 1.5)
    
    # Pool sustainability factor
    sustainability_factor = np.clip(reward_pool_ratio, 0.5, 1.2)
    
    # Velocity-based adjustment to control inflation
    velocity_adjustment = 1 / (1 + max(0, token_velocity - 12) * 0.1)  # Dampen rewards when velocity is high
    
    # Calculate final reward with inflation control
    adjusted_reward = base_reward * (
        utility_factor *
        network_factor *
        maturity_factor *
        supply_adjustment *
        sustainability_factor *
        velocity_adjustment *
        inflation_dampener  # Apply inflation control last
    )
    
    # Circuit breakers for extreme cases
    if inflation_rate > 0.15:  # 15% inflation threshold
        adjusted_reward *= 0.5  # Emergency 50% reduction
    
    return adjusted_reward

def calculate_dynamic_decay(
    base_decay_rate,
    engagement_score,
    value_trend,
    reward_pool_ratio,
    market_maturity,
    price_ratio,
    supply_ratio,
    token_velocity,
    inflation_rate
):
    """Calculate dynamic decay rate with stabilization mechanisms"""
    # Base decay adjustment
    base_adjustment = np.clip(1 + value_trend, 0.5, 1.5)
    
    # Supply pressure adjustment
    supply_pressure = np.clip(supply_ratio, 0.5, 2.0)
    supply_adjustment = 1 / supply_pressure
    
    # Price stability adjustment
    price_stability = 1 / (1 + abs(price_ratio - 1) * 2)
    
    # Velocity-based adjustment
    velocity_adjustment = np.clip(12 / max(token_velocity, 1), 0.5, 1.5)
    
    # Pool sustainability adjustment
    sustainability_adjustment = np.clip(reward_pool_ratio, 0.5, 2.0)
    
    # Market maturity impact
    maturity_adjustment = 1 + market_maturity * 0.5  # Higher decay in mature markets
    
    # Inflation control
    inflation_adjustment = np.clip(1 + (inflation_rate - 0.05) * 2, 0.5, 2.0)
    
    # Calculate effective decay rate with stability controls
    effective_decay = base_decay_rate * (
        base_adjustment *
        supply_adjustment *
        price_stability *
        velocity_adjustment *
        sustainability_adjustment *
        maturity_adjustment *
        inflation_adjustment
    )
    
    # Circuit breakers
    if inflation_rate > 0.1:  # High inflation
        effective_decay *= 1.5  # Increase decay to combat inflation
    elif supply_ratio < 0.8:  # Supply shortage
        effective_decay *= 0.5  # Reduce decay to preserve supply
    
    return np.clip(effective_decay, base_decay_rate * 0.5, base_decay_rate * 2.0)

def calculate_price_stabilization(
    current_price,
    target_price,
    inflation_rate,
    token_velocity,
    market_maturity,
    supply_ratio,
    reward_decay_rate
):
    """Calculate price stabilization factors"""
    # Price deviation from target
    price_deviation = abs(current_price / target_price - 1)
    
    # Stability thresholds
    stability_thresholds = {
        "low": 0.05,    # 5% deviation
        "medium": 0.15, # 15% deviation
        "high": 0.25    # 25% deviation
    }
    
    # Calculate stabilization pressure
    if price_deviation <= stability_thresholds["low"]:
        stability_pressure = 0  # No intervention needed
    elif price_deviation <= stability_thresholds["medium"]:
        stability_pressure = 0.5  # Moderate intervention
    else:
        stability_pressure = 1.0  # Strong intervention
    
    # Adjust for market maturity
    stability_pressure *= (1 + market_maturity)  # Stronger stabilization in mature markets
    
    # Velocity adjustment
    velocity_factor = np.clip(12 / max(token_velocity, 1), 0.5, 1.5)
    
    # Supply adjustment
    supply_factor = np.clip(1 / supply_ratio, 0.5, 1.5)
    
    # Decay impact
    decay_factor = np.clip(1 / (1 + reward_decay_rate * 5), 0.5, 1.5)
    
    return {
        'stability_pressure': stability_pressure,
        'velocity_factor': velocity_factor,
        'supply_factor': supply_factor,
        'decay_factor': decay_factor
    }

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

    # Initialize user segments with different behaviors
    user_segments = {
        "power": {
            "proportion": 0.1,  # 10% of users
            "activity_multiplier": 2.0,  # 2x average activity
            "premium_likelihood": 0.8,  # 80% likely to use premium
            "churn_resistance": 2.0,  # 2x more resistant to churn
            "token_holding_time": 3  # Holds tokens for 3 months on average
        },
        "regular": {
            "proportion": 0.3,  # 30% of users
            "activity_multiplier": 1.0,  # Average activity
            "premium_likelihood": 0.4,  # 40% likely to use premium
            "churn_resistance": 1.0,  # Average churn resistance
            "token_holding_time": 2  # Holds tokens for 2 months on average
        },
        "casual": {
            "proportion": 0.6,  # 60% of users
            "activity_multiplier": 0.5,  # Half average activity
            "premium_likelihood": 0.1,  # 10% likely to use premium
            "churn_resistance": 0.5,  # Half as resistant to churn
            "token_holding_time": 1  # Holds tokens for 1 month on average
        }
    }

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

    # Initialize simulation variables
    users = base_users
    total_tokens_earned = 0
    total_tokens_spent = 0
    total_tokens_burned = 0
    total_tokens_staked = 0
    treasury_tokens = reward_pool_size
    circulating_supply = 0
    locked_ratio = 0
    churn_ratio = 0
    engagement_factor = 1.0
    utility_price_floor = initial_token_price * 0.5
    value_per_user = 0
    total_value_created = 0
    effective_supply = reward_pool_size
    total_demand = 0
    
    # Initialize price impact components
    supply_demand_impact = 0
    scarcity_impact = 0
    utility_impact = 0
    market_impact = 0
    sentiment_impact = 0
    network_price_impact = 0
    momentum_impact = 0
    trend_following = 0
    
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
    monthly_results = []
    price_history = [initial_token_price]
    sentiment_history = [initial_market_sentiment]
    price_change_history = [0]
    transaction_volume_history = [0]
    
    # Initialize staking metrics
    staked_tokens_by_duration = {  # Track staking by lock period
        "30_days": 0,
        "90_days": 0,
        "180_days": 0,
        "365_days": 0
    }
    staking_rewards_by_duration = {  # Different APRs for different lock periods
        "30_days": staking_apr * 0.5,
        "90_days": staking_apr * 0.8,
        "180_days": staking_apr * 1.2,
        "365_days": staking_apr * 1.5
    }
    unstaking_penalties = {  # Penalties for early unstaking
        "30_days": 0.1,
        "90_days": 0.15,
        "180_days": 0.2,
        "365_days": 0.25
    }
    
    # Initialize liquidity metrics
    order_book_depth = {
        "bids": [],  # List of (price, amount) tuples
        "asks": []
    }
    liquidity_depth = token_price * total_tokens_earned * 0.1  # Initial liquidity
    slippage_factor = 0.01  # 1% base slippage
    
    # Initialize holder distribution
    token_holder_distribution = {
        "whales": {"count": 0, "total_tokens": 0},      # >1% of supply
        "large": {"count": 0, "total_tokens": 0},       # 0.1-1% of supply
        "medium": {"count": 0, "total_tokens": 0},      # 0.01-0.1% of supply
        "small": {"count": 0, "total_tokens": 0}        # <0.01% of supply
    }

    # Initialize market phase based on initial market saturation
    initial_market_saturation = base_users / total_addressable_market
    market_phase = (
        "early" if initial_market_saturation < 0.1 else
        "growth" if initial_market_saturation < 0.5 else
        "mature"
    )

    # Initialize segment counts
    segment_counts = {
        segment: int(users * data["proportion"]) 
        for segment, data in user_segments.items()
    }

    # Define initial seed value for zero-start case
    initial_seed = max(10, int(total_users_target * 0.001))  # Start with 0.1% of target or minimum 10 users

    # Ensure targets don't exceed TAM
    total_users_target = min(total_users_target, total_addressable_market)
    carrying_capacity = min(carrying_capacity or total_addressable_market, total_addressable_market)

    # Calculate maximum possible growth rate based on TAM
    if base_users == 0:
        max_growth_rate = min(2.0, growth_rate)  # Cap initial growth at 200%
    else:
        max_growth_rate = (total_addressable_market / base_users) ** (1/months) - 1
        max_growth_rate = min(max_growth_rate, 2.0)  # Cap at 200% to prevent unrealistic growth

    effective_growth_rate = min(growth_rate, max_growth_rate)
    
    log_message(f"Base Users: {base_users}", debug_only=True)
    log_message(f"TAM: {total_addressable_market}", debug_only=True)
    log_message(f"Months: {months}", debug_only=True)
    log_message(f"Max Growth Rate: {max_growth_rate}", debug_only=True)
    log_message(f"Input Growth Rate: {growth_rate}", debug_only=True)
    log_message(f"Effective Growth Rate: {effective_growth_rate}", debug_only=True)

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
        users,
        token_price,
        initial_token_price,
        price_elasticity,
        inactivity_rate,
        user_growth_rate,    # Added parameter
        engagement_score     # Added parameter
    ):
        """Calculate burn dynamics with comprehensive user metrics coupling"""
        # Base burn calculation with price elasticity
        price_ratio = token_price / initial_token_price
        elastic_burn_rate = burn_rate * (price_ratio ** price_elasticity)
        base_burn = monthly_token_spending * elastic_burn_rate
        
        # Supply impact calculation
        circulating_supply = max(1, total_tokens_earned - total_tokens_burned - total_tokens_staked)
        burn_supply_ratio = total_tokens_burned / total_tokens_earned if total_tokens_earned > 0 else 0
        
        # Enhanced burn pressure calculation with comprehensive user metrics
        burn_pressure = (
            (1 + burn_supply_ratio * 2.5) *              
            (1 + token_velocity / 10) *                  
            (1 + (platform_revenue / (users * token_price) if users > 0 and token_price > 0 else 0)) *
            (1 + inactivity_rate * 1.5) *               # Inactivity impact
            (1 - user_growth_rate * 0.5) *              # Growth dampens burn
            (2 - engagement_score)                       # Low engagement increases burn
        )
        
        # Market phase multipliers with user-metric sensitivity
        phase_multipliers = {
            "early": {
                "base": 2.0,
                "growth_sensitivity": 0.7,    # Higher growth impact
                "engagement_sensitivity": 1.2  # Higher engagement impact
            },
            "growth": {
                "base": 1.5,
                "growth_sensitivity": 0.5,
                "engagement_sensitivity": 1.0
            },
            "mature": {
                "base": 1.0,
                "growth_sensitivity": 0.3,
                "engagement_sensitivity": 0.8
            }
        }[market_phase]
        
        # Calculate effective multiplier
        effective_multiplier = (
            phase_multipliers["base"] *
            (1 - user_growth_rate * phase_multipliers["growth_sensitivity"]) *
            (2 - engagement_score * phase_multipliers["engagement_sensitivity"])
        )
        
        # Calculate effective burn with enhanced dynamics
        effective_burn = (
            base_burn * 
            effective_multiplier * 
            burn_pressure * 
            (1 - market_maturity * 0.4) *  
            (1 + inactivity_rate)
        )
        
        # Dynamic circuit breakers based on user metrics
        supply_threshold = 0.2 * (1 + engagement_score * 0.5)  # Higher threshold for engaged users
        if circulating_supply < total_tokens_earned * supply_threshold:
            reduction_factor = 0.6 * (1 + user_growth_rate)  # Less reduction if growing
            effective_burn *= reduction_factor
            log_message(f"Supply threshold reached, reducing burn by {(1-reduction_factor)*100:.0f}%", debug_only=True)
        
        return {
            'burn_amount': effective_burn,
            'burn_pressure': burn_pressure,
            'effective_burn_rate': elastic_burn_rate * effective_multiplier,
            'supply_impact': -effective_burn / circulating_supply if circulating_supply > 0 else 0,
            'user_metrics_impact': (2 - engagement_score) * (1 + inactivity_rate) * (1 - user_growth_rate * 0.5)
        }

    def calculate_price_impact(
        token_price,
        initial_token_price,
        burn_rate,
        inactivity_rate,
        price_elasticity,
        market_sentiment,
        market_volatility,
        burn_dynamics,
        user_metrics,
        supply_metrics,
        market_phase,
        user_growth_rate,    # Added parameter
        engagement_score     # Added parameter
    ):
        """Calculate comprehensive price impact with stronger user metrics coupling"""
        # Enhanced burn impact with user metrics
        burn_price_impact = burn_dynamics['supply_impact'] * (
            1 + burn_rate * 3 +
            burn_dynamics['burn_pressure'] * 0.8 +
            burn_dynamics['user_metrics_impact'] * 0.5  # Added direct user metrics impact
        )
        
        # Enhanced activity impact with growth consideration
        activity_impact = (
            (1 - inactivity_rate) * (
                engagement_score * 0.6 +
                user_metrics['growth_rate'] * 0.4 +
                user_metrics['value_per_user'] * 0.3
            ) * (1 + max(0, user_growth_rate) * 0.5)  # Growth bonus
        )
        
        # Network effect impact
        network_impact = np.clip(
            user_growth_rate * engagement_score * (1 - inactivity_rate),
            -0.3, 0.5
        )
        
        # Enhanced supply impact with user metrics
        supply_impact = (
            supply_metrics['scarcity_ratio'] * 0.5 +
            supply_metrics['velocity_ratio'] * 0.4 +
            supply_metrics['staking_ratio'] * 0.4
        ) * (1 + engagement_score * 0.3)  # Engagement multiplier
        
        # Phase-specific weights with enhanced user metrics sensitivity
        weights = {
            "early": {
                "burn": 0.35,
                "activity": 0.25,
                "elasticity": 0.15,
                "sentiment": 0.15,
                "supply": 0.1,
                "network": 0.2     # Added network effect weight
            },
            "growth": {
                "burn": 0.3,
                "activity": 0.3,
                "elasticity": 0.15,
                "sentiment": 0.1,
                "supply": 0.15,
                "network": 0.15
            },
            "mature": {
                "burn": 0.25,
                "activity": 0.25,
                "elasticity": 0.2,
                "sentiment": 0.15,
                "supply": 0.15,
                "network": 0.1
            }
        }[market_phase]
        
        # Calculate final price impact with network effect
        price_impact = (
            burn_price_impact * weights["burn"] +
            activity_impact * weights["activity"] +
            elastic_impact * weights["elasticity"] +
            sentiment_volatility * weights["sentiment"] +
            supply_impact * weights["supply"] +
            network_impact * weights["network"]  # Added network effect
        )
        
        return price_impact

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

    def generate_order_book_side(base_price, liquidity_depth, side):
        """Generate synthetic order book entries for one side."""
        num_levels = 10
        orders = []
        
        if side == "bid":
            # Generate bid orders below base price
            for i in range(num_levels):
                price = base_price * (1 - 0.01 * i)  # 1% price steps
                amount = liquidity_depth * (1 - 0.1 * i)  # Decreasing liquidity
                orders.append((price, amount))
        else:
            # Generate ask orders above base price
            for i in range(num_levels):
                price = base_price * (1 + 0.01 * i)  # 1% price steps
                amount = liquidity_depth * (1 - 0.1 * i)  # Decreasing liquidity
                orders.append((price, amount))
        
        return orders

    def calculate_price_impact(transaction_size, liquidity_depth, slippage):
        """Calculate price impact of a transaction."""
        impact = (transaction_size / liquidity_depth) * slippage
        return np.clip(impact, 0, 0.5)  # Max 50% impact

    def update_holder_distribution(distribution, total_supply, burned, staked):
        """Update token holder distribution metrics."""
        circulating = total_supply - burned - staked
        if circulating <= 0:
            return
        
        # Define thresholds
        whale_threshold = circulating * 0.01
        large_threshold = circulating * 0.001
        medium_threshold = circulating * 0.0001
        
        # Reset counts
        for category in distribution:
            distribution[category] = {"count": 0, "total_tokens": 0}
        
        # Simulate holder distribution based on power law
        total_holders = int(np.sqrt(circulating))  # Simplified holder count
        for i in range(total_holders):
            # Power law distribution of holdings
            tokens = circulating * (1 / (i + 1)) ** 1.5
            
            if tokens >= whale_threshold:
                distribution["whales"]["count"] += 1
                distribution["whales"]["total_tokens"] += tokens
            elif tokens >= large_threshold:
                distribution["large"]["count"] += 1
                distribution["large"]["total_tokens"] += tokens
            elif tokens >= medium_threshold:
                distribution["medium"]["count"] += 1
                distribution["medium"]["total_tokens"] += tokens
            else:
                distribution["small"]["count"] += 1
                distribution["small"]["total_tokens"] += tokens

    def calculate_holder_concentration(distribution):
        """Calculate holder concentration metric."""
        total_tokens = sum(cat["total_tokens"] for cat in distribution.values())
        if total_tokens <= 0:
            return 0
        
        # Weight larger holders more heavily
        concentration = (
            distribution["whales"]["total_tokens"] * 0.5 +
            distribution["large"]["total_tokens"] * 0.3 +
            distribution["medium"]["total_tokens"] * 0.15 +
            distribution["small"]["total_tokens"] * 0.05
        ) / total_tokens
        
        return np.clip(concentration, 0, 1)

    def calculate_price_bounds(current_price, maturity, concentration, slippage, floor_price):
        """Calculate dynamic price bounds based on market conditions."""
        # Base volatility bounds
        base_range = 0.2  # 20% base range
        
        # Adjust range based on market conditions
        maturity_factor = 1 - (maturity * 0.5)  # Less volatile as market matures
        concentration_factor = 1 + (concentration * 0.5)  # More volatile with high concentration
        slippage_factor = 1 + slippage
        
        effective_range = base_range * maturity_factor * concentration_factor * slippage_factor
        
        return {
            "lower": max(floor_price, current_price * (1 - effective_range)),
            "upper": current_price * (1 + effective_range)
        }

    # Calculate user growth with logistic constraints and market saturation
    def calculate_user_growth(current_users, growth_rate, carrying_capacity, market_saturation, time_step):
        """Calculate constrained user growth using logistic function and market dynamics"""
        # Base logistic growth
        remaining_capacity = max(0, carrying_capacity - current_users)
        saturation_factor = remaining_capacity / carrying_capacity if carrying_capacity > 0 else 0
        
        # Market saturation dampening
        market_dampening = 1 - (market_saturation ** 2)  # Quadratic slowdown as market saturates
        
        # Growth rate adjustment
        effective_growth = growth_rate * saturation_factor * market_dampening
        
        # Calculate new users with constraints
        user_increase = current_users * effective_growth * (1 - current_users / carrying_capacity)
        return max(0, min(user_increase, remaining_capacity))

    # Initialize treasury and emission metrics
    treasury_tokens = reward_pool_size
    emission_rate = 0.05  # 5% base emission rate
    target_supply = reward_pool_size * (1 + market_maturity)
    
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
        
        # Calculate market metrics with enhanced dynamics
        market_saturation = min(users / total_addressable_market, 1.0)
        market_maturity = np.sqrt(market_saturation)
        
        # Calculate activity and engagement metrics
        total_activity = sum(
            count * user_segments[segment]["activity_multiplier"]
            for segment, count in segment_counts.items()
        )
        expected_activity = users * np.mean([data["activity_multiplier"] for data in user_segments.values()])
        activity_ratio = total_activity / expected_activity if expected_activity > 0 else 0
        
        # Calculate inactivity impact
        inactive_users = sum(
            count * (1 - user_segments[segment]["activity_multiplier"] * activity_ratio)
            for segment, count in segment_counts.items()
        )
        inactivity_ratio = inactive_users / users if users > 0 else 0
        
        # Calculate engagement score
        engagement_score = (
            (1 - inactivity_ratio) *                     # Active user ratio
            activity_ratio *                             # Activity level
            (1 - churn_ratio) *                         # Retention impact
            (1 + premium_adoption * 0.5)                # Premium user boost
        )
        
        # Adjust token velocity by engagement
        effective_velocity = token_velocity * np.clip(engagement_score, 0.5, 1.5)
        
        # Update effective supply with engagement impact
        effective_supply = circulating_supply * (
            1 - (locked_ratio * 0.5) *                   # Impact of locked tokens
            (1 + market_maturity) *                      # Market maturity effect
            (1 + network_effect) *                       # Network effects
            (2 - engagement_score)                       # Lower engagement increases effective supply
        )
        
        # Calculate engagement-adjusted demand
        active_users_demand = users * value_per_user * effective_velocity * engagement_score
        speculative_demand = token_purchase_amount if token_price <= token_purchase_threshold else 0
        staking_demand = total_tokens_staked * staking_apr / 12  # Monthly staking rewards demand
        total_demand = active_users_demand + speculative_demand + staking_demand
        
        # Update price components with engagement
        engagement_price_impact = (engagement_score - 1) * 0.3  # ±30% max impact
        
        price_change = (
            supply_demand_impact * 0.25 +                # 25% supply-demand
            scarcity_impact * 0.15 +                     # 15% scarcity
            utility_impact * 0.15 +                      # 15% utility
            engagement_price_impact * 0.15 +             # 15% engagement
            market_impact * 0.1 +                        # 10% market depth
            sentiment_impact * 0.1 +                     # 10% sentiment
            network_price_impact * 0.05 +                # 5% network growth
            momentum_impact * 0.025 +                    # 2.5% momentum
            trend_following * 0.025                      # 2.5% trend
        )
        
        # Adjust rewards based on engagement
        adjusted_reward = adjusted_reward * np.clip(engagement_score, 0.3, 1.2)
        
        # Update burn rate with inactivity impact
        effective_burn_rate *= (1 + (inactivity_ratio * 0.5))  # Increase burn when inactive
        
        # Calculate market metrics
        market_saturation = min(users / total_addressable_market, 1.0)
        market_maturity = np.sqrt(market_saturation)
        
        # Calculate stagnation metrics
        activity_change = (total_activity - previous_activity) / previous_activity if 'previous_activity' in locals() and previous_activity > 0 else 0
        transaction_change = (current_transaction_volume - previous_transaction_volume) / previous_transaction_volume if previous_transaction_volume > 0 else 0
        growth_stagnation = 1 - np.clip(user_growth_rate / effective_growth_rate if effective_growth_rate > 0 else 0, 0, 1)
        
        # Calculate composite stagnation score
        stagnation_score = np.clip(
            (growth_stagnation * 0.4 +                  # 40% weight to growth stagnation
             abs(min(activity_change, 0)) * 0.3 +       # 30% weight to activity decline
             abs(min(transaction_change, 0)) * 0.3      # 30% weight to transaction decline
            ),
            0, 1
        )
        
        # Calculate dynamic market confidence
        market_confidence = (
            (1 - stagnation_score) *                    # Reduced confidence during stagnation
            (1 + network_effect * 0.5) *                # Network effects boost confidence
            (1 + market_maturity * 0.3) *               # Mature markets more resilient
            (1 - competitive_pressure * 0.2)            # Competition reduces confidence
        )
        
        # Update price components with stagnation effects
        stagnation_price_impact = -stagnation_score * 0.5  # Up to 50% negative impact
        confidence_price_impact = (market_confidence - 1) * 0.3  # ±30% impact
        
        # Calculate price change with stagnation effects
        price_change = (
            supply_demand_impact * 0.2 +                # 20% supply-demand
            scarcity_impact * 0.15 +                    # 15% scarcity
            utility_impact * 0.15 +                     # 15% utility
            sentiment_impact * 0.2 +                    # 20% sentiment (increased)
            volatility_price_impact * 0.15 +            # 15% volatility
            network_price_impact * 0.1 +                # 10% network
            momentum_impact * 0.05                      # 5% momentum
        )
        
        # Apply volatility-based price bounds
        volatility_bound_multiplier = (
            1 + market_volatility * (
                2 if sentiment < 1 else    # Amplified impact in bear market
                1 if sentiment < 3 else    # Normal impact in neutral market
                0.5) *                     # Reduced impact in bull market
            (1 + abs(price_trend))                  # Trend amplification
        )

        price_bounds = calculate_price_bounds(
            current_price=token_price,
            maturity=market_maturity,
            concentration=calculate_holder_concentration(token_holder_distribution),
            slippage=slippage_factor * volatility_bound_multiplier,
            floor_price=utility_price_floor
        )
        
        # Update token price with stagnation-aware bounds
        token_price = np.clip(
            token_price * (1 + price_change),
            price_bounds["lower"],
            price_bounds["upper"]
        )
        
        # Adjust rewards for stagnation
        stagnation_reward_modifier = (1 - stagnation_score * 0.7)  # Up to 70% reduction
        adjusted_reward *= stagnation_reward_modifier
        
        # Update burn rate with stagnation
        stagnation_burn_multiplier = 1 + (stagnation_score * 0.5)  # Up to 50% increase
        effective_burn_rate *= stagnation_burn_multiplier
        
        # Store metrics for next iteration
        previous_activity = total_activity
        previous_transaction_volume = current_transaction_volume
        
        # Calculate network growth and value metrics
        user_growth_rate = (users / base_users - 1) if base_users > 0 else 1
        growth_acceleration = user_growth_rate - (previous_user_growth_rate if 'previous_user_growth_rate' in locals() else 0)
        previous_user_growth_rate = user_growth_rate
        
        # Calculate per-user value metrics
        value_per_user = (current_transaction_volume / users) if users > 0 else 0
        network_value_coefficient = np.sqrt(market_saturation) * (1 + value_per_user / initial_search_fee)
        
        # Dynamic reward scaling based on network growth
        network_scale_factor = np.log1p(user_growth_rate) * network_value_coefficient
        base_reward_scalar = np.clip(1 + network_scale_factor, 0.5, 3.0)
        
        # Calculate supply pressure with network scaling
        supply_pressure = (
            (total_tokens_earned - total_tokens_burned) / 
            (reward_pool_size * (1 + market_saturation))  # Scale with network size
        ) if reward_pool_size > 0 else 1

        supply_pressure = np.clip(supply_pressure, 0.5, 2.0)

        # Calculate value metrics for reward adjustment
        value_per_user = (current_transaction_volume / users) if users > 0 else 0
        value_trend = (value_per_user / previous_value_per_user - 1) if 'previous_value_per_user' in locals() and previous_value_per_user > 0 else 0
        previous_value_per_user = value_per_user
        
        # Calculate reward pool sustainability
        reward_pool_ratio = treasury_tokens / reward_pool_size
        reward_runway_months = treasury_tokens / (adjusted_reward * users) if adjusted_reward * users > 0 else float('inf')
        
        # Calculate adaptive reward with all stabilization factors
        adjusted_reward = calculate_adaptive_reward(
            base_reward=initial_reward,
            engagement_score=engagement_score,
            value_per_user=value_per_user,
            network_effect=network_effect,
            market_maturity=market_maturity,
            reward_pool_ratio=reward_pool_ratio,
            price_ratio=token_price / initial_token_price,
            supply_ratio=supply_pressure,
            token_velocity=token_velocity
        )
        
        # Calculate dynamic decay with inflation control
        effective_decay_rate = calculate_dynamic_decay(
            base_decay_rate=reward_decay_rate,
            engagement_score=engagement_score,
            value_trend=value_trend,
            reward_pool_ratio=reward_pool_ratio,
            market_maturity=market_maturity,
            price_ratio=token_price / initial_token_price,
            supply_ratio=supply_pressure,
            token_velocity=token_velocity,
            inflation_rate=inflation_rate
        )
        
        # Apply controlled reward adjustment
        reward_change = (adjusted_reward - reward) * 0.2  # 20% max monthly adjustment
        reward = np.clip(
            reward + reward_change,
            reward * 0.8,  # Max 20% decrease
            reward * 1.2   # Max 20% increase
        )
        
        # Apply dynamic decay with circuit breakers
        if inflation_rate > 0.5:  # >50% annual inflation
            reward *= 0.7  # Emergency 30% reduction
            log_message("Warning: High inflation detected, implementing emergency reward reduction", debug_only=True)

        if supply_pressure > 2.0:  # Severe oversupply
            reward *= 0.8  # Emergency 20% reduction
            log_message("Warning: Supply pressure detected, reducing rewards", debug_only=True)

        # Final bounds check with dynamic limits
        max_reward = initial_reward * (
            1.0 +                          # Base cap
            network_effect * 0.3 +         # Network boost (max 30%)
            market_maturity * 0.2 -        # Maturity reduction (max 20%)
            inflation_rate                 # Inflation penalty
        )

        min_reward = initial_reward * 0.1  # Hard floor at 10% of initial

        reward = np.clip(reward, min_reward, max_reward)
        
        # Calculate reward contribution with smoothing
        reward_contribution = calculate_reward_contribution(
            reward=reward,
            users=users,
            treasury_tokens=treasury_tokens,
            previous_contribution=previous_reward_contribution if 'previous_reward_contribution' in locals() else None,
            market_maturity=market_maturity,
            engagement_score=engagement_score
        )

        # Store for next iteration
        previous_reward_contribution = reward_contribution

        # Update treasury with smoothed contribution
        treasury_tokens = max(0, treasury_tokens - reward_contribution)

        # Calculate market conditions for treasury management
        market_conditions = {
            'demand': total_demand,
            'supply': effective_supply,
            'demand_pressure': np.clip(total_demand / effective_supply - 1, 0, 1) if effective_supply > 0 else 0
        }
        
        # Calculate controlled token emission
        emission_amount = calculate_token_emission(
            current_supply=total_tokens_earned - total_tokens_burned,
            target_supply=target_supply,
            emission_rate=emission_rate,
            market_maturity=market_maturity,
            network_growth=user_growth_rate if 'user_growth_rate' in locals() else 0
        )
        
        # Manage treasury releases
        treasury_release = manage_treasury_tokens(
            treasury_balance=treasury_tokens,
            circulating_supply=total_tokens_earned - total_tokens_burned,
            market_conditions=market_conditions
        )
        
        # Update token supply metrics
        treasury_tokens = max(0, treasury_tokens - treasury_release)
        total_tokens_earned += treasury_release
        
        # Apply controlled emission
        if emission_amount > 0:
            treasury_tokens += emission_amount
        
        # Store metrics for next iteration
        previous_activity = total_activity
        previous_transaction_volume = current_transaction_volume
        
        # Calculate final treasury value for this month
        treasury_change, new_treasury = manage_treasury_changes(
            current_treasury=treasury_tokens,
            reward_pool_size=reward_pool_size,
            revenue_change=search_revenue + premium_revenue,
            emission_change=emission_amount if 'emission_amount' in locals() else 0,
            market_maturity=market_maturity,
            previous_changes=treasury_changes if 'treasury_changes' in locals() else None,
            token_velocity=token_velocity,
            market_phase=market_phase
        )

        # Single point of truth for treasury update
        treasury_tokens = new_treasury
        
        # Store change for next iteration
        treasury_changes = treasury_changes[-3:] if 'treasury_changes' in locals() else []
        treasury_changes.append(treasury_change)
        
        # Single point of data collection AFTER all updates
        monthly_data = {
            'month': current_month,
            'treasury_tokens': float(treasury_tokens),  # Ensure numeric type
            'platform_revenue': float(search_revenue + premium_revenue),
            'token_price': float(token_price),
            'users': int(users),
            'market_phase': str(market_phase),
            'market_sentiment': float(sentiment),
            'engagement_score': float(engagement_score),
            'total_tokens_burned': float(total_tokens_burned),
            'total_tokens_staked': float(total_tokens_staked),
            'circulating_supply': float(circulating_supply),
            'token_velocity': float(token_velocity),
            'inflation_rate': float(inflation_rate)
        }
        
        # Debug verification
        log_message(f"Month {current_month}:", debug_only=True)
        log_message(f"  Treasury: {treasury_tokens:.2f}", debug_only=True)
        log_message(f"  Revenue: {search_revenue + premium_revenue:.2f}", debug_only=True)
        log_message(f"  Change: {treasury_change:.2f}", debug_only=True)
        
        # Single append with copy to prevent reference issues
        monthly_results.append(monthly_data.copy())
    
    # Create DataFrame once at the end
    results_df = pd.DataFrame(monthly_results)
    
    # Verify data integrity
    log_message(f"Total months: {len(results_df)}", debug_only=True)
    log_message(f"Treasury range: {results_df['treasury_tokens'].min():.2f} to {results_df['treasury_tokens'].max():.2f}", debug_only=True)
    
    return results_df

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

def collect_monthly_metrics(state_dict):
    """Collect monthly metrics in a clean, consistent way"""
    return {
        'month': state_dict['current_month'],
        'treasury_tokens': float(state_dict['treasury_tokens']),
        'platform_revenue': float(state_dict['search_revenue'] + state_dict['premium_revenue']),
        'token_price': float(state_dict['token_price']),
        'users': int(state_dict['users']),
        'market_phase': str(state_dict['market_phase']),
        'market_sentiment': float(state_dict['sentiment']),
        'engagement_score': float(state_dict['engagement_score']),
        'total_tokens_burned': float(state_dict['total_tokens_burned']),
        'total_tokens_staked': float(state_dict['total_tokens_staked']),
        'circulating_supply': float(state_dict['circulating_supply']),
        'token_velocity': float(state_dict['token_velocity']),
        'inflation_rate': float(state_dict['inflation_rate'])
    }

def main():
    """Main function to run simulation and create visualization"""
    try:
        # Get simulation inputs
        simulation_params = get_simulation_inputs()
        
        # Run simulation with fresh parameters
        results = simulate_tokenomics(*simulation_params)
        
        # Clean and verify results
        results = results.drop_duplicates(subset=['month'], keep='last')
        results = results.sort_values('month')
        
        # Verify data integrity
        log_message(f"Data shape: {results.shape}", debug_only=True)
        log_message(f"Unique months: {results['month'].nunique()}", debug_only=True)
        log_message(f"Treasury range: {results['treasury_tokens'].min():.2f} to {results['treasury_tokens'].max():.2f}", debug_only=True)
        
        return results
        
    except Exception as e:
        log_message(f"Simulation error: {str(e)}", debug_only=False)
        raise

if __name__ == "__main__":
    try:
        results = main()
        log_message("Simulation completed successfully", debug_only=True)
    except Exception as e:
        log_message(f"Error in simulation: {str(e)}", debug_only=False)

def calculate_spending_metrics(
    users,
    user_segments,
    segment_counts,
    search_fee,
    premium_fee,
    previous_spending,
    previous_premium_spending,
    market_maturity,
    engagement_score,
    value_per_user,
    market_phase,
    max_monthly_growth=0.3  # Reduced from 0.5 to 0.3
):
    """Calculate controlled spending metrics with growth caps"""
    # Base spending calculation per segment
    base_spending = sum(
        segment_counts[segment] * 
        user_segments[segment]["activity_multiplier"] * 
        search_fee
        for segment in user_segments
    )
    
    # Base premium spending calculation
    base_premium = sum(
        segment_counts[segment] * 
        user_segments[segment]["premium_likelihood"] * 
        premium_fee
        for segment in user_segments
    )
    
    # Phase-specific growth limits (tightened)
    max_growth = {
        "early": max_monthly_growth,          # 30% max in early
        "growth": max_monthly_growth * 0.7,   # 21% max in growth
        "mature": max_monthly_growth * 0.4    # 12% max in mature
    }[market_phase]
    
    # Adjust growth limit based on market maturity (more aggressive reduction)
    adjusted_max_growth = max_growth * (1 - market_maturity * 0.7)  # Up to 70% reduction
    
    # Calculate growth-constrained spending
    if previous_spending > 0:
        max_spending = previous_spending * (1 + adjusted_max_growth)
        base_spending = min(base_spending, max_spending)
    
    if previous_premium_spending > 0:
        max_premium = previous_premium_spending * (1 + adjusted_max_growth)
        base_premium = min(base_premium, max_premium)
    
    # Apply engagement and value modifiers (tightened ranges)
    spending_modifier = np.clip(
        engagement_score * (1 + value_per_user / search_fee - 1) * 0.3,  # Reduced from 0.5
        0.7,  # Increased min from 0.5
        1.3   # Reduced max from 1.5
    )
    
    # Calculate final spending with modifiers
    monthly_spending = base_spending * spending_modifier
    premium_spending = base_premium * spending_modifier
    
    return monthly_spending, premium_spending

def calculate_revenue_metrics(
    monthly_spending,
    premium_spending,
    token_price,
    market_maturity,
    previous_revenue,
    treasury_tokens,
    reward_pool_size,
    max_revenue_growth=0.25
):
    """Calculate revenue with smoothing and diminishing returns"""
    
    # Input smoothing with exponential moving average
    def smooth_input(current, previous, alpha=0.3):
        if previous is None:
            return current
        return alpha * current + (1 - alpha) * previous
    
    # Smooth spending inputs
    smoothed_monthly = smooth_input(monthly_spending, previous_revenue * 0.7 if previous_revenue else None)
    smoothed_premium = smooth_input(premium_spending, previous_revenue * 0.3 if previous_revenue else None)
    
    # Convert spending to base token demand
    base_search_tokens = smoothed_monthly / token_price if token_price > 0 else 0
    base_premium_tokens = smoothed_premium / token_price if token_price > 0 else 0
    
    # Apply diminishing returns instead of bulk discounts
    def apply_diminishing_returns(token_amount):
        # Logarithmic scaling for diminishing returns
        if token_amount <= 0:
            return 0
        base = 1000  # Base unit for scaling
        scale_factor = np.log10(1 + token_amount / base)
        return token_amount * (1 / (1 + scale_factor * 0.2))  # Max 20% reduction
    
    # Calculate effective token revenue with diminishing returns
    search_revenue = apply_diminishing_returns(base_search_tokens)
    premium_revenue = apply_diminishing_returns(base_premium_tokens)
    
    # Gradual treasury capacity adjustment
    total_revenue = search_revenue + premium_revenue
    max_treasury_revenue = treasury_tokens * 0.03  # Reduced from 0.05
    
    if total_revenue > max_treasury_revenue:
        # More gradual adjustment
        adjustment_factor = 0.2  # 20% adjustment per step
        target_ratio = max_treasury_revenue / total_revenue
        current_ratio = 1.0
        ratio = current_ratio + (target_ratio - current_ratio) * adjustment_factor
        
        search_revenue *= ratio
        premium_revenue *= ratio
    
    return search_revenue, premium_revenue

def calculate_user_behavior_adjustment(
    sentiment,
    volatility,
    market_phase,
    user_segment,
    price_trend,
    token_price,
    initial_token_price,
    value_per_user,
    network_effect
):
    """Calculate user behavior adjustments based on market conditions"""
    # Base sentiment impact varies by user segment
    sentiment_sensitivity = {
        "power": 0.8,     # Power users more resilient
        "regular": 1.0,   # Regular users standard sensitivity
        "casual": 1.2     # Casual users more sensitive
    }[user_segment]
    
    # Volatility impact varies by user segment
    volatility_sensitivity = {
        "power": 0.7,     # Power users handle volatility better
        "regular": 1.0,   # Regular users standard sensitivity
        "casual": 1.3     # Casual users more risk-averse
    }[user_segment]
    
    # Calculate price stress
    price_stress = abs(token_price / initial_token_price - 1) * (1 + volatility * 2)
    
    # Base activity modification
    activity_modifier = (
        (1 + (sentiment - 1) * sentiment_sensitivity) *     # Sentiment impact
        (1 - volatility * volatility_sensitivity) *         # Volatility impact
        (1 + price_trend * 0.3) *                          # Price trend impact
        (1 - price_stress * 0.2)                           # Price stress impact
    )
    
    # Staking behavior modification
    staking_modifier = (
        (1 + max(0, sentiment - 1) * 1.5) *                # Positive sentiment boosts staking
        (1 - volatility * 0.8) *                           # High volatility reduces staking
        (1 + network_effect * 0.3)                         # Network effect encourages staking
    )
    
    # Token holding time modification
    holding_modifier = (
        (1 + max(0, sentiment - 1) * 1.2) *                # Positive sentiment increases holding
        (1 - volatility * 1.2) *                           # Volatility reduces holding time
        (1 + value_per_user / initial_token_price * 0.3)   # Value creation encourages holding
    )
    
    # Calculate churn risk
    churn_risk = (
        max(0, (1 - sentiment) * 0.3) +                    # Low sentiment increases churn
        volatility * 0.4 +                                 # Volatility increases churn
        max(0, -price_trend) * 0.3                        # Negative price trend increases churn
    )
    
    # Market phase adjustments
    phase_multipliers = {
        "early": {
            "activity": 1.2,    # Higher activity tolerance in early phase
            "staking": 0.8,     # Lower staking in early phase
            "holding": 0.7,     # Lower holding time in early phase
            "churn": 1.2        # Higher churn risk in early phase
        },
        "growth": {
            "activity": 1.0,    # Standard activity in growth phase
            "staking": 1.2,     # Higher staking in growth phase
            "holding": 1.0,     # Standard holding time in growth phase
            "churn": 1.0        # Standard churn risk in growth phase
        },
        "mature": {
            "activity": 0.8,    # Lower activity tolerance in mature phase
            "staking": 1.5,     # Higher staking in mature phase
            "holding": 1.3,     # Higher holding time in mature phase
            "churn": 0.8        # Lower churn risk in mature phase
        }
    }[market_phase]
    
    return {
        'activity_modifier': activity_modifier * phase_multipliers["activity"],
        'staking_modifier': staking_modifier * phase_multipliers["staking"],
        'holding_modifier': holding_modifier * phase_multipliers["holding"],
        'churn_risk': min(1.0, churn_risk * phase_multipliers["churn"])
    }

def calculate_market_sentiment_impact(
    price_trend,
    volatility,
    market_phase,
    engagement_score,
    network_effect,
    holder_concentration,
    competitive_pressure,
    price_ratio,
    value_per_user,
    initial_value_per_user
):
    """Calculate comprehensive market sentiment impact"""
    # Base sentiment components
    price_sentiment = np.clip(1 + price_trend * 2, 0.5, 2.0)
    value_sentiment = np.clip(value_per_user / initial_value_per_user, 0.5, 2.0)
    network_sentiment = np.clip(1 + network_effect, 0.5, 2.0)
    
    # Volatility impact
    volatility_impact = -volatility * (
        2.0 if price_trend < 0 else    # Stronger impact in downtrend
        1.0 if price_trend == 0 else   # Normal impact in sideways market
        0.5                            # Reduced impact in uptrend
    )
    
    # Concentration impact (high concentration increases volatility)
    concentration_impact = -holder_concentration * volatility * 1.5
    
    # Competitive pressure impact
    competition_impact = -competitive_pressure * (1 - network_effect * 0.5)
    
    # Phase-specific sentiment modifiers
    phase_modifiers = {
        "early": {
            "price_weight": 0.3,      # Lower price influence
            "value_weight": 0.3,      # Equal value influence
            "network_weight": 0.4,     # Higher network influence
            "volatility_mult": 1.5,    # Higher volatility impact
            "concentration_mult": 1.2,  # Higher concentration impact
            "competition_mult": 0.8     # Lower competition impact
        },
        "growth": {
            "price_weight": 0.35,     # Balanced price influence
            "value_weight": 0.35,     # Balanced value influence
            "network_weight": 0.3,     # Balanced network influence
            "volatility_mult": 1.0,    # Standard volatility impact
            "concentration_mult": 1.0,  # Standard concentration impact
            "competition_mult": 1.0     # Standard competition impact
        },
        "mature": {
            "price_weight": 0.4,      # Higher price influence
            "value_weight": 0.4,      # Higher value influence
            "network_weight": 0.2,     # Lower network influence
            "volatility_mult": 0.7,    # Lower volatility impact
            "concentration_mult": 0.8,  # Lower concentration impact
            "competition_mult": 1.2     # Higher competition impact
        }
    }[market_phase]
    
    # Calculate weighted sentiment
    base_sentiment = (
        price_sentiment * phase_modifiers["price_weight"] +
        value_sentiment * phase_modifiers["value_weight"] +
        network_sentiment * phase_modifiers["network_weight"]
    )
    
    # Apply market condition impacts
    market_impacts = (
        volatility_impact * phase_modifiers["volatility_mult"] +
        concentration_impact * phase_modifiers["concentration_mult"] +
        competition_impact * phase_modifiers["competition_mult"]
    )
    
    # Calculate final sentiment with engagement influence
    final_sentiment = base_sentiment * (1 + market_impacts) * engagement_score
    
    # Apply circuit breakers for extreme conditions
    if volatility > 0.5:  # High volatility
        final_sentiment *= 0.7  # Strong sentiment dampening
    if holder_concentration > 0.7:  # High concentration
        final_sentiment *= 0.8  # Moderate sentiment dampening
    
    return np.clip(final_sentiment, 0.2, 3.0)  # Allow for more extreme sentiment range

def calculate_reward_contribution(
    reward,
    users,
    treasury_tokens,
    previous_contribution=None,
    market_maturity=0,
    engagement_score=1.0,
    previous_users=None,
    previous_activity=None
):
    """Calculate reward contribution with pre-emptive smoothing"""
    # Smooth user growth first
    def smooth_growth(current, previous, alpha=0.3):
        if previous is None:
            return current
        max_change = previous * 0.2  # Max 20% change per period
        raw_change = current - previous
        smoothed_change = np.clip(raw_change, -max_change, max_change)
        return previous + smoothed_change * alpha
    
    # Smooth users before reward calculation
    smoothed_users = smooth_growth(users, previous_users)
    
    # Calculate base contribution with smoothed users
    base_contribution = reward * smoothed_users
    
    # Progressive scaling based on user count
    scale_factor = np.clip(1 / np.log10(max(smoothed_users, 10)), 0.3, 1.0)
    base_contribution *= scale_factor
    
    # Apply exponential smoothing to final contribution
    def smooth_contribution(current, previous, alpha=0.2):  # Reduced alpha
        if previous is None:
            return current
        return alpha * current + (1 - alpha) * previous
    
    # Smooth the contribution
    smoothed_contribution = smooth_contribution(base_contribution, previous_contribution)
    
    # Market maturity and engagement impact
    maturity_factor = 1 - (market_maturity * 0.5)  # Increased dampening
    engagement_factor = 0.8 + (engagement_score * 0.2)  # Reduced range
    
    # Calculate final contribution with stricter caps
    max_contribution = min(
        treasury_tokens * 0.03,  # Reduced from 0.05
        smoothed_contribution * maturity_factor * engagement_factor
    )
    
    return max_contribution

def calculate_activity_metrics(
    segment_counts,
    user_segments,
    previous_activity=None
):
    """Calculate smoothed activity metrics"""
    # Calculate raw activity
    raw_activity = sum(
        count * user_segments[segment]["activity_multiplier"]
        for segment, count in segment_counts.items()
    )
    
    # Smooth activity changes
    def smooth_activity(current, previous, alpha=0.2):
        if previous is None:
            return current
        max_change = previous * 0.15  # Max 15% change
        raw_change = current - previous
        smoothed_change = np.clip(raw_change, -max_change, max_change)
        return previous + smoothed_change * alpha
    
    smoothed_activity = smooth_activity(raw_activity, previous_activity)
    
    # Calculate activity ratio with smoothing
    expected_activity = sum(
        count * 1.0  # Base multiplier
        for count in segment_counts.values()
    )
    
    activity_ratio = smoothed_activity / expected_activity if expected_activity > 0 else 0
    
    return smoothed_activity, activity_ratio