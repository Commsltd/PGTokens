import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def get_token_metrics(results):
    """
    Calculate comprehensive token performance metrics.
    
    Args:
        results: DataFrame containing simulation results
    
    Returns:
        dict: Token performance metrics
    """
    current_price = results['Token Price'].iloc[-1]
    initial_price = results['Token Price'].iloc[0]
    price_change = ((current_price / initial_price) - 1) * 100
    max_price = results['Token Price'].max()
    min_price = results['Token Price'].min()
    price_volatility = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0

    return {
        "current_price": current_price,
        "initial_price": initial_price,
        "price_change": price_change,
        "max_price": max_price,
        "min_price": min_price,
        "price_volatility": price_volatility,
    }

def get_user_metrics(results):
    """
    Calculate comprehensive user metrics.
    
    Args:
        results: DataFrame with simulation results
    
    Returns:
        dict: User metrics
    """
    current_users = results['Users'].iloc[-1]
    initial_users = results['Users'].iloc[0]
    user_growth = ((current_users / initial_users) - 1) * 100
    peak_users = results['Users'].max()
    
    return {
        'current_users': current_users,
        'initial_users': initial_users,
        'user_growth': user_growth,
        'peak_users': peak_users
    }

def calculate_reserve_health(current_reserve, initial_reserve):
    """
    Calculate reserve health status with detailed warnings and visual indicators.
    
    Args:
        current_reserve: Current reward pool balance
        initial_reserve: Initial reward pool size
    
    Returns:
        tuple: (health_status, reserve_ratio)
    """
    ratio = current_reserve / initial_reserve
    
    if ratio >= 0.50:
        return "Healthy 🟢", ratio
    elif ratio >= 0.25:
        return "Warning 🟡", ratio
    else:
        return "Critical 🔴", ratio

def calculate_reserve_metrics(results, initial_reserve):
    """
    Calculate comprehensive reserve metrics including time in critical levels and deficits.
    
    Args:
        results: DataFrame with simulation results
        initial_reserve: Initial reward pool size
    
    Returns:
        dict: Comprehensive reserve metrics
    """
    warning_threshold = initial_reserve * 0.5
    critical_threshold = initial_reserve * 0.25
    
    # Calculate time spent in each state
    total_months = len(results)
    months_in_warning = len(results[
        (results['Reward Pool'] <= warning_threshold) & 
        (results['Reward Pool'] > critical_threshold)
    ])
    months_in_critical = len(results[
        (results['Reward Pool'] <= critical_threshold) & 
        (results['Reward Pool'] > 0)
    ])
    months_in_deficit = len(results[results['Reward Pool'] <= 0])
    
    # Calculate cumulative deficit
    deficit_mask = results['Reward Pool'] < 0
    cumulative_deficit = abs(results.loc[deficit_mask, 'Reward Pool'].sum())
    
    # Calculate recovery metrics
    recovery_count = 0
    current_state = "healthy"
    for _, row in results.iterrows():
        if row['Reward Pool'] <= critical_threshold and current_state != "critical":
            current_state = "critical"
        elif row['Reward Pool'] > warning_threshold and current_state == "critical":
            current_state = "healthy"
            recovery_count += 1
    
    # Calculate volatility
    reserve_volatility = results['Reward Pool'].std() / initial_reserve * 100
    
    # Calculate minimum reserve ratio
    min_reserve_ratio = results['Reward Pool'].min() / initial_reserve
    
    return {
        'warning_threshold': warning_threshold,
        'critical_threshold': critical_threshold,
        'months_in_warning': months_in_warning,
        'months_in_critical': months_in_critical,
        'months_in_deficit': months_in_deficit,
        'warning_percentage': (months_in_warning / total_months) * 100,
        'critical_percentage': (months_in_critical / total_months) * 100,
        'deficit_percentage': (months_in_deficit / total_months) * 100,
        'cumulative_deficit': cumulative_deficit,
        'recovery_count': recovery_count,
        'reserve_volatility': reserve_volatility,
        'min_reserve_ratio': min_reserve_ratio
    }

def calculate_monthly_burn_metrics(current_reserve, initial_reserve, months):
    """
    Calculate burn rate metrics and projected depletion timeline.
    
    Args:
        current_reserve: Current reward pool balance
        initial_reserve: Initial reward pool size
        months: Number of months simulated
    
    Returns:
        dict: Burn rate metrics including monthly burn, burn rate percentage, and months remaining
    """
    # Calculate total burned and monthly burn rate
    total_burned = initial_reserve - current_reserve
    monthly_burn = total_burned / months if months > 0 else 0
    
    # Calculate months until depletion at current burn rate
    months_remaining = current_reserve / monthly_burn if monthly_burn > 0 else float('inf')
    
    return {
        'monthly_burn': monthly_burn,
        'burn_rate_pct': (monthly_burn / initial_reserve) * 100,
        'months_remaining': months_remaining
    }

# Create a number input for total users with scientific notation support
def format_number(num):
    """Format large numbers into Thousand, Million format"""
    if num >= 1e6:
        return f"{num/1e6:.1f} Million"
    elif num >= 1e3:
        return f"{num/1e3:.1f} Thousand"
    return str(num)

def format_currency(amount):
    """Format large currency numbers into Thousand, Million, Billion format"""
    if amount >= 1e9:
        return f"${amount/1e9:.1f} Billion"
    elif amount >= 1e6:
        return f"${amount/1e6:.1f} Million"
    elif amount >= 1e3:
        return f"${amount/1e3:.1f} Thousand"
    return f"${amount:.2f}"

# Define the Plotly chart creation function
def create_plotly_chart(df, x, y, title, y_label):
    fig = px.line(
        df,
        x=x,
        y=y,
        title=title,
        labels={x: x} if isinstance(y, str) else {},
        template="plotly_dark",
        width=800,
        height=450,
        color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']  # Multiple distinct colors
    )
    
    # Update line styling (only width, let color be handled by color_discrete_sequence)
    if isinstance(y, str):
        fig.update_traces(line=dict(width=2.5))
    else:
        for trace in fig.data:
            trace.update(line=dict(width=2.5))
    
    # Update layout for better formatting
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            font=dict(size=20),
            xanchor='center',
            yanchor='top'
        ),
        margin=dict(l=60, r=30, t=50, b=60),
        yaxis_title=y_label,
        xaxis_title="Month",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        showlegend=True if isinstance(y, list) else False,
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.1)'
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.1)'
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    
    return fig

def create_dual_axis_chart(df):
    # Create figure with secondary y-axis
    fig = px.line(
        df, 
        x="Month", 
        y="Token Price",
        title="Token Price & Platform Revenue Over Time",
        template="plotly_dark",
        width=800,
        height=450
    )
    
    # Add Platform Revenue on secondary y-axis
    fig.add_scatter(
        x=df["Month"],
        y=df["Platform Revenue ($)"],
        name="Platform Revenue",
        yaxis="y2",
        line=dict(color="#00CC96", width=2.5)  # Green color for revenue
    )
    
    # Update layout for dual axes
    fig.update_layout(
        title=dict(
            text="Token Price & Platform Revenue Over Time",
            x=0.5,
            y=0.95,
            font=dict(size=20),
            xanchor='center',
            yanchor='top'
        ),
        yaxis=dict(
            title="Token Price ($)",
            titlefont=dict(color="#636EFA"),  # Blue color for token price
            tickfont=dict(color="#636EFA"),
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.1)'
        ),
        yaxis2=dict(
            title="Platform Revenue ($)",
            titlefont=dict(color="#00CC96"),  # Green color for revenue
            tickfont=dict(color="#00CC96"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        xaxis=dict(
            title="Month",
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.1)'
        ),
        margin=dict(l=60, r=60, t=50, b=60),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)"
        ),
        hovermode='x unified'
    )
    
    return fig

# --- Functions for Token Price and Market Sentiment ---

def calculate_token_price(
    base_price,
    total_tokens_issued,
    total_tokens_spent,
    price_elasticity,
    market_sentiment,
):
    if total_tokens_issued <= 0:
        return base_price

    # Ensure supply_demand_ratio is positive
    supply_demand_ratio = max(0.001, total_tokens_spent / total_tokens_issued)

    price = base_price * (supply_demand_ratio**price_elasticity) * market_sentiment
    return max(0.01, price)

def simulate_market_sentiment(initial_sentiment, volatility, trend):
    """
    Simulates market sentiment changes with controlled volatility.
    
    Args:
        initial_sentiment: Current market sentiment value
        volatility: How much sentiment can change (clipped between 1-5%)
        trend: Long-term trend direction
    
    Returns:
        new_sentiment: Updated market sentiment value
    """
    sentiment = initial_sentiment
    sentiment += np.random.normal(0, volatility) + trend
    return max(0.1, min(2.0, sentiment))  # Bound sentiment between 0.1 and 2.0

# Add this function before the simulate_tokenomics function
def vesting_schedule(month, total_vested, vest_duration):
    """
    Vesting schedule with cliff period: returns how many tokens vest this month.
    
    Args:
        month: Current month in simulation (0-based)
        total_vested: Total tokens that will be vested over entire schedule
        vest_duration: Number of months for vesting
    
    Returns:
        tokens_vested_this_month: Number of tokens vesting in current month
    """
    if month >= vest_duration:
        return 0  # No vesting after duration
    elif month >= vest_duration // 4:  # Introduce a cliff
        return (total_vested / vest_duration) * 0.8  # Vest 80% post-cliff
    else:
        return (total_vested / vest_duration) * 0.2  # Vest 20% during cliff

# Add this function before the simulate_tokenomics function
def logistic_growth(current_month, carrying_capacity, initial_base_users, growth_steepness, midpoint):
    """
    Implements an S-curve growth model using the logistic function.
    
    Args:
        current_month: Current month in simulation (0-based)
        carrying_capacity: Maximum possible users (usually TAM)
        initial_base_users: Starting number of users
        growth_steepness: Controls how steep the S-curve is (typically 0.1-0.5)
        midpoint: Month at which growth is ~50% of capacity
    
    Returns:
        user_count: Projected number of users at this point in time
    """
    # Ensure parameters are within valid ranges
    growth_steepness = np.clip(growth_steepness, 0.1, 0.5)
    carrying_capacity = max(carrying_capacity, initial_base_users * 2)
    midpoint = max(1, midpoint)
    
    # Calculate logistic growth
    growth_factor = -growth_steepness * (current_month - midpoint)
    user_count = carrying_capacity / (1 + np.exp(growth_factor))
    
    # Ensure we don't go below initial users or above carrying capacity
    user_count = max(initial_base_users, min(user_count, carrying_capacity))
    
    return user_count

# --- Function to Create Sliders with Customizable Ranges and Error Handling---
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

# --- Main Simulation Function ---

def apply_shock_event(current_params, event_info):
    """
    Applies shock event modifications to simulation parameters.
    Returns modified parameters without altering the originals.
    
    Args:
        current_params: dict of current simulation parameters
        event_info: dict containing shock event modifications
    
    Returns:
        dict: Modified parameters for this month
    """
    # Create a copy of current parameters to modify
    params = current_params.copy()
    
    # Apply growth rate modifications
    if "growth_rate_boost" in event_info:
        params["growth_rate"] = params.get("growth_rate", 0) + event_info["growth_rate_boost"]
        params["growth_rate"] = max(0, min(params["growth_rate"], 1.0))  # Bound between 0-100%

    # Apply sentiment modifications
    if "sentiment_shift" in event_info:
        params["market_sentiment"] = params.get("market_sentiment", 1.0) + event_info["sentiment_shift"]
        params["market_sentiment"] = max(0.1, min(params["market_sentiment"], 2.0))  # Bound between 0.1-2.0

    # Apply inactivity rate modifications
    if "inactivity_spike" in event_info:
        params["inactivity_rate"] = params.get("inactivity_rate", 0.05) + event_info["inactivity_spike"]
        params["inactivity_rate"] = max(0.01, min(params["inactivity_rate"], 0.15))  # Bound between 1-15%

    # Apply token price shock
    if "price_shock" in event_info:
        params["token_price_multiplier"] = event_info["price_shock"]

    # Apply reward pool modifications
    if "reward_pool_change" in event_info:
        params["reward_pool_modifier"] = event_info["reward_pool_change"]

    return params

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
    new_customers_per_user,
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
    transaction_fee_rate,
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
    # Ensure the new carrying_capacity has a default value
    if carrying_capacity is None:
        carrying_capacity = total_addressable_market

    # Ensure targets don't exceed TAM
    total_users_target = min(total_users_target, total_addressable_market)
    base_users = max(1000, int(total_users_target * 0.1))
    
    # Calculate maximum possible growth rate based on TAM
    max_growth_rate = (total_addressable_market / base_users) ** (1/months) - 1
    effective_growth_rate = min(growth_rate, max_growth_rate)
    
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

    user_segments = {
        "power": {
            "proportion": 0.1,
            "contribution_multiplier": 5,
            "lookup_multiplier": 2,
            "premium_adoption_multiplier": 2,
        },
        "regular": {
            "proportion": 0.6,
            "contribution_multiplier": 1,
            "lookup_multiplier": 1,
            "premium_adoption_multiplier": 1,
        },
        "casual": {
            "proportion": 0.3,
            "contribution_multiplier": 0.2,
            "lookup_multiplier": 0.5,
            "premium_adoption_multiplier": 0.5,
        },
    }

    monthly_results = []

    # Initialize segment counts based on initial user distribution
    segment_counts = {
        segment: int(users * data["proportion"]) for segment, data in user_segments.items()
    }

    # Initialize user token balances
    user_token_balances = {
        "power": segment_counts["power"] * initial_token_price,
        "regular": segment_counts["regular"] * initial_token_price,
        "casual": segment_counts["casual"] * initial_token_price,
    }

    # Cap each competitor's attractiveness at 3%
    competitor_attractiveness = [min(0.03, attractiveness) for attractiveness in competitor_attractiveness]

    # Initialize simulation parameters dictionary
    current_params = {
        "growth_rate": effective_growth_rate,
        "market_sentiment": initial_market_sentiment,
        "inactivity_rate": inactivity_rate,
        "token_price_multiplier": 1.0,
        "reward_pool_modifier": 1.0
    }

    # Validate shock events structure
    if shock_events is not None:
        # Ensure all event months are within simulation range
        shock_events = {
            month: events for month, events in shock_events.items()
            if 1 <= month <= months
        }

    # --- Simulation Loop ---
    for month in range(months):
        current_month = month + 1
        
        # Apply shock events if any exist for this month
        if shock_events and current_month in shock_events:
            # Get modified parameters for this month
            modified_params = apply_shock_event(current_params, shock_events[current_month])
            
            # Apply modified parameters
            effective_growth_rate = modified_params["growth_rate"]
            market_sentiment = modified_params["market_sentiment"]
            inactivity_rate = modified_params["inactivity_rate"]
            
            # Apply token price shock if present
            if modified_params["token_price_multiplier"] != 1.0:
                token_price *= modified_params["token_price_multiplier"]
            
            # Apply reward pool modification if present
            if modified_params["reward_pool_modifier"] != 1.0:
                reward_pool *= modified_params["reward_pool_modifier"]

        # Track shock event occurrences
        shock_event_description = None
        if shock_events and current_month in shock_events:
            event_info = shock_events[current_month]
            descriptions = []
            if "growth_rate_boost" in event_info:
                descriptions.append(f"Growth {'+' if event_info['growth_rate_boost'] > 0 else ''}{event_info['growth_rate_boost']*100:.1f}%")
            if "sentiment_shift" in event_info:
                descriptions.append(f"Sentiment {'+' if event_info['sentiment_shift'] > 0 else ''}{event_info['sentiment_shift']:.2f}")
            if "inactivity_spike" in event_info:
                descriptions.append(f"Churn +{event_info['inactivity_spike']*100:.1f}%")
            if "price_shock" in event_info:
                descriptions.append(f"Price x{event_info['price_shock']:.2f}")
            if "reward_pool_change" in event_info:
                descriptions.append(f"Rewards x{event_info['reward_pool_change']:.2f}")
            shock_event_description = " | ".join(descriptions)

        # 1. VESTING (updated logic)
        if vest_duration > 0 and total_vested_tokens > 0:
            newly_vested = vesting_schedule(month, total_vested_tokens, vest_duration)
            total_tokens_earned += newly_vested

        # 2. USER GROWTH & COMPETITION (modified for logistic)
        if logistic_enabled:
            # We compute the logistic-based user count for this month
            logistic_user_count = logistic_growth(
                current_month=month,
                carrying_capacity=carrying_capacity,
                initial_base_users=base_users,
                growth_steepness=growth_steepness,
                midpoint=midpoint_month
            )
            new_users = int(logistic_user_count - users) if logistic_user_count > users else 0
        else:
            # Your original code path: purely linear-based approach
            new_users = int(users * effective_growth_rate)

        # Apply competition and churn effects
        churned_users = int(users * np.clip(inactivity_rate, 0.03, 0.08))

        # Include competitor effects with safeguards
        competitor_churn = 0
        for i in range(num_competitors):
            competitor_effect = np.clip(
                competitor_attractiveness[i] * (1 - token_price / initial_token_price),
                0,
                0.03  # Cap maximum user loss to any single competitor at 3%
            )
            competitor_churn += int(users * competitor_effect)
        
        churned_users += competitor_churn

        # Update total users with bounds checking
        previous_users = users
        users += new_users - churned_users
        users = max(base_users * 0.5, min(users, carrying_capacity))  # Never drop below 50% of base or exceed capacity
        
        # Calculate actual growth rate for metrics
        actual_growth_rate = (users - previous_users) / previous_users if previous_users > 0 else 0

        # Update segment counts each month
        segment_counts = {
            segment: int(users * data["proportion"]) for segment, data in user_segments.items()
        }

        # 3. Line-Item Contributions
        # Calculate total contributions across all segments
        total_contributions = sum(segment_counts[segment] * user_segments[segment]["contribution_multiplier"] for segment in user_segments)

        # Distribute rewards proportionally based on contributions
        monthly_rewards = 0
        reward *= 1 - np.clip(reward_decay_rate, 0.01, 0.05)  # Clip decay rate between 1-5%

        # Calculate and clip monthly contributions
        monthly_contributions = total_contributions  # This should be your existing contributions calculation
        if contribution_cap:
            monthly_contributions = min(monthly_contributions, users * np.clip(contribution_cap, 200, 500))

        # Use monthly_contributions for reward calculations
        if monthly_contributions > 0:  # Prevent division by zero
            for segment in user_segments:
                proportional_reward = (segment_counts[segment] * user_segments[segment]["contribution_multiplier"]) / total_contributions
                segment_rewards = reward_pool * proportional_reward
                monthly_rewards += segment_rewards
                reward_pool -= segment_rewards

        total_tokens_earned += monthly_rewards
        reward *= 1 - reward_decay_rate

        # 4. Search Activity (Lookups)
        monthly_lookups = 0
        for segment, data in user_segments.items():
            monthly_lookups += (
                segment_counts[segment]
                * customers_per_user
                * (lookup_frequency / 12)
                * data["lookup_multiplier"]
            )
        monthly_lookups += users * new_customers_per_user
        monthly_spending = monthly_lookups * search_fee
        total_tokens_spent += monthly_spending

        # Adjust lookup frequency based on token price and market sentiment
        lookup_frequency_adjustment = 1 + (initial_token_price - token_price) * 0.1 * market_sentiment
        lookup_frequency *= max(0.5, min(2.0, lookup_frequency_adjustment))  # Bound adjustments

        # Check if users need to purchase more tokens
        for segment, balance in user_token_balances.items():
            if balance < token_purchase_threshold:
                tokens_to_buy = token_purchase_amount
                user_token_balances[segment] += tokens_to_buy
                platform_revenue += tokens_to_buy * token_sale_price

        # 5. Premium Feature Spending
        premium_spending = 0
        for segment, data in user_segments.items():
            premium_spending += (
                segment_counts[segment]
                * premium_adoption
                * 10
                * data["premium_adoption_multiplier"]
            )
        total_tokens_spent += premium_spending

        # Update user token balances after spending
        for segment, data in user_segments.items():
            user_token_balances[segment] -= (
                segment_counts[segment]
                * premium_adoption
                * 10
                * data["premium_adoption_multiplier"]
            )

        # 6. Token Burning
        tokens_to_burn = total_tokens_earned * min(burn_rate, 0.03)  # Cap burn rate at 3%
        total_tokens_burned += tokens_to_burn
        total_tokens_earned -= tokens_to_burn

        # 7. Staking (simplified)
        monthly_staking_apr = staking_apr / 12
        staking_rewards = total_tokens_staked * monthly_staking_apr
        reward_pool -= staking_rewards
        total_tokens_earned += staking_rewards
        reward_pool = max(0, reward_pool)  # Prevent negative reward pool

        # Assume some users stake tokens each month (simplified logic)
        new_tokens_staked = users * (token_price / initial_token_price) * 0.1  # More staking if token price is high
        total_tokens_staked += new_tokens_staked
        total_tokens_earned -= new_tokens_staked
        total_tokens_staked = max(0, total_tokens_staked)  # Prevent negative staked tokens

        # 8. Transaction Fees
        transaction_fees = (monthly_spending * transaction_fee_rate) * token_price
        platform_revenue += transaction_fees

        # Use transaction fees for the reward pool and burning (example)
        reward_pool += transaction_fees * 0.50
        total_tokens_burned += transaction_fees * 0.25 * (1 / token_price)

        # 9. Market Sentiment
        market_sentiment = simulate_market_sentiment(
            market_sentiment, 
            np.clip(market_volatility, 0.01, 0.05),  # Clip volatility between 1-5%
            0.01  # Slight positive trend
        )

        # 10. Token Price
        token_price = calculate_token_price(
            initial_token_price,
            total_tokens_earned,
            total_tokens_spent,
            price_elasticity,
            market_sentiment,
        )

        # 11. Platform Revenue (moved up from previous position 12)
        platform_revenue += monthly_spending * token_price * 0.75

        # --- Store Monthly Results ---
        monthly_results.append(
            {
                "Month": current_month,
                "Users": users,
                "Token Price": token_price,
                "Market Sentiment": market_sentiment,
                "Tokens Earned": total_tokens_earned,
                "Tokens Spent": total_tokens_spent,
                "Tokens Burned": total_tokens_burned,
                "Net Token Balance": total_tokens_earned - total_tokens_spent,
                "Platform Revenue ($)": platform_revenue,
                "Search Fee": search_fee,
                "Reward": reward,
                "Power Users": segment_counts["power"],
                "Regular Users": segment_counts["regular"],
                "Casual Users": segment_counts["casual"],
                "Total Tokens Staked": total_tokens_staked,
                "Reward Pool": reward_pool,
                "Token Sales Revenue": platform_revenue,
                "Actual Growth Rate": actual_growth_rate,
                "Target Growth Rate": effective_growth_rate if not logistic_enabled else None,
                "Distance to Carrying Capacity": carrying_capacity - users if logistic_enabled else None,
                "Shock Event": shock_event_description,
            }
        )

    return pd.DataFrame(monthly_results)

# --- Streamlit UI ---
st.set_page_config(page_title="PG Tokenomics Simulator", layout="wide")
st.title("PG Tokenomics Simulator")

# Add at the beginning of your sidebar parameters, before other user metrics
st.sidebar.header("🎯 Market Size Parameters")

# Total Addressable Market (TAM) with slider
total_addressable_market = st.sidebar.number_input(
    "Total Addressable Market (Users)",
    min_value=10_000,
    max_value=1_000_000_000,
    value=1_000_000,
    help="The maximum possible number of users for your platform"
)

# Format the display
formatted_tam = format_number(total_addressable_market)
st.sidebar.markdown(f"""
    💡 **Market Size Breakdown:**
    - Total Addressable Market: **{formatted_tam} users**
    - Initial Target: **{format_number(total_addressable_market * 0.1)} users** (10% of TAM)
""")

# User Growth Target with slider
total_users_target = create_slider_with_range(
    "User Growth Target",
    default_min=1000,
    default_max=total_addressable_market,
    default_value=min(100_000, total_addressable_market),
    step=1000,
    key_prefix="target",
    help_text="Your target number of users (cannot exceed Total Addressable Market)"
)

# Show percentage of TAM
target_percentage = (total_users_target / total_addressable_market) * 100
st.sidebar.markdown(f"Target represents **{target_percentage:.1f}%** of total market")

# --- Sidebar for Custom Slider Ranges ---
st.sidebar.title("Customize Slider Ranges")
st.sidebar.write("Adjust the minimum and maximum values for the sliders.")

# --- Input Sliders Organized into Sections ---

# --- Section 1: User & Growth Parameters ---
st.sidebar.header("1. User & Growth Parameters")

# Function to create sliders with dynamic range and error handling
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

# Example of using the slider with a smoothing factor
def calculate_user_growth(users, growth_rate, smoothing_factor=0.1):
    # Apply a smoothing factor to reduce sensitivity
    adjusted_growth_rate = growth_rate * (1 - smoothing_factor) + smoothing_factor
    new_users = int(users * adjusted_growth_rate)
    return new_users

# User Growth Rate Slider
growth_rate = create_slider_with_range(
    "User Growth Rate (%)",
    0.00,
    20.00,
    6.00,  # Default 6% monthly growth
    0.01,
    format="%.2f",
    key_prefix="s1",
    help_text="The percentage by which the user base grows each month (recommended: 5-7%)."
)

# Inactivity Rate Slider
inactivity_rate = create_slider_with_range(
    "Inactivity Rate (% of Users per Month)",
    0.00,
    20.00,
    4.00,  # Default 4% churn
    0.01,
    format="%.2f",
    key_prefix="s1",
    help_text="The percentage of users that become inactive each month (target: <5%)."
)

# Initial User Base Slider
base_users = create_slider_with_range(
    "Initial User Base",
    100,
    total_addressable_market,
    int(total_addressable_market * 0.125),  # 12.5% of TAM
    100,
    key_prefix="s1",
    help_text="The starting number of users (recommended: 10-15% of TAM)."
)

# --- Section 2: Token & Reward Parameters ---
st.sidebar.header("2. Token & Reward Parameters")

initial_reward = create_slider_with_range(
    "Initial Reward per Line Item (Tokens)",
    0.01,
    10.0,
    0.35,  # Default 0.35 tokens
    0.01,
    key_prefix="s2",
    help_text="Tokens awarded per line item (recommended: 0.25-0.5)."
)

# Reward Decay Rate Slider
reward_decay_rate = create_slider_with_range(
    "Reward Decay Rate (%)",
    0.00,
    20.00,
    1.50,  # Default 1.5%
    0.01,
    format="%.2f",
    key_prefix="s2",
    help_text="Monthly reward reduction rate (recommended: 1-2%)."
)

burn_rate = create_slider_with_range(
    "Token Burn Rate (%)",
    0.00,
    10.00,
    4.00,  # Default 4%
    0.01,
    format="%.2f",
    key_prefix="s2",
    help_text="Percentage of tokens burned (recommended: 3-5%)."
)

staking_apr = create_slider_with_range(
    "Staking APR (%)",
    0.0,
    20.0,
    7.5,  # Default 7.5%
    0.1,
    format="%.2f",
    key_prefix="s2",
    help_text="Annual percentage rate for staking (recommended: 5-10%)."
)

reward_pool_size = st.sidebar.number_input("Initial Reward Pool Size", value=1000000)

# --- Section 3:  Platform Activity Parameters ---

st.sidebar.header("3. Platform Activity Parameters")

line_items_per_customer = create_slider_with_range(
    "Line Items per Customer",
    1,
    5000,
    75,  # Default 75 items
    1,
    key_prefix="s3",
    help_text="Monthly line items per customer (recommended: 50-100)."
)

contribution_cap = create_slider_with_range(
    "Contribution Cap (Line Items per User per Month)",
    1,
    10000,
    750,  # Default 750 items
    1,
    key_prefix="s3",
    help_text="Maximum monthly contributions per user (recommended: 500-1000)."
)

initial_lookup_frequency = create_slider_with_range(
    "Initial Lookups per Customer per Year",
    1,
    100,
    9,  # Default 9 lookups
    1,
    key_prefix="s3",
    help_text="Annual customer lookup frequency (recommended: 6-12)."
)

initial_premium_adoption = create_slider_with_range(
    "Initial Premium Adoption Rate (% of Users)",
    0.00,
    1.00,
    0.25,  # Default 25%
    0.01,
    key_prefix="s3",
    help_text="Starting premium feature adoption rate (recommended: 20-30%)."
)

customers_per_user = create_slider_with_range(
    "Customers per User",
    1,
    1000,
    50,
    1,
    key_prefix="s3",
    help_text="The average number of customers associated with each user.",
)

new_customers_per_user = create_slider_with_range(
    "New Customers per User (per month)",
    1,
    100,
    7,  # Default 7 new customers
    1,
    key_prefix="s3",
    help_text="Average new customers acquired per user monthly (recommended: 5-10)."
)

initial_search_fee = create_slider_with_range(
    "Initial Search Fee (Tokens)",
    0.1,
    50.0,
    5.0,
    0.1,
    key_prefix="s3",
    help_text="The initial number of tokens required to perform a search.",
)

transaction_fee_rate = create_slider_with_range(
    "Transaction Fee Rate (%)",
    0.00,
    20.00,
    4.00,  # Default 4%
    0.01,
    format="%.2f",
    key_prefix="s3",
    help_text="Fee percentage per transaction (recommended: 3-5%)."
)

# Add sliders for token purchase parameters
token_purchase_threshold = create_slider_with_range(
    "Token Purchase Threshold (Tokens)",
    0.0,
    100.0,
    7.5,  # Default 7.5 tokens
    0.1,
    key_prefix="s3",
    help_text="Token balance triggering purchases (recommended: 5-10)."
)

token_purchase_amount = create_slider_with_range(
    "Token Purchase Amount (Tokens)",
    1.0,
    100.0,
    15.0,  # Default 15 tokens
    1.0,
    key_prefix="s3",
    help_text="Tokens purchased when below threshold (recommended: 10-20)."
)

token_sale_price = create_slider_with_range(
    "Token Sale Price ($ per Token)",
    0.01,
    10.0,
    1.0,
    0.01,
    key_prefix="s3",
    help_text="The price at which tokens are sold to users."
)

# --- Section 4: Market Parameters ---
st.sidebar.header("4. Market Parameters")

initial_token_price = create_slider_with_range(
    "Initial Token Price ($)",
    0.01,
    10.0,
    1.0,
    0.01,
    key_prefix="s4",
    help_text="The starting price of the token in USD."
)

price_elasticity = create_slider_with_range(
    "Price Elasticity",
    0.1,
    2.0,
    0.4,  # Default 0.4
    0.1,
    key_prefix="s4",
    help_text="Token price sensitivity to supply/demand (recommended: 0.3-0.5)."
)

initial_market_sentiment = create_slider_with_range(
    "Initial Market Sentiment",
    0.5,
    1.5,
    1.0,  # Default neutral
    0.1,
    key_prefix="s4",
    help_text="Starting market sentiment (1.0 is neutral)."
)

market_volatility = create_slider_with_range(
    "Market Volatility",
    0.01,
    0.5,
    0.15,  # Default 0.15
    0.01,
    format="%.2f",
    key_prefix="s4",
    help_text="Market sentiment volatility (recommended: 0.1-0.2)."
)

market_trend = create_slider_with_range(
    "Market Trend",
    -0.1,
    0.1,
    0.0,  # Default neutral
    0.01,
    key_prefix="s4",
    help_text="Long-term market sentiment trend (0.0 is neutral)."
)

# --- Competition Parameters ---
num_competitors = st.sidebar.number_input("Number of Competitors", 0, 10, 3)
competitor_growth_rates = [0.04] * num_competitors  # Default 4% growth
competitor_attractiveness = [0.02] * num_competitors  # Default 0.02 attractiveness

# --- Simulation Parameters ---
months = st.sidebar.number_input("Simulation Duration (Months)", 12, 120, 36)

# Temporarily disable shock events for testing
shock_events = None

# --- Run Simulation ---
results = simulate_tokenomics(
    initial_reward=initial_reward,
    initial_search_fee=initial_search_fee,
    growth_rate=growth_rate,
    line_items_per_customer=line_items_per_customer,
    initial_lookup_frequency=initial_lookup_frequency,
    reward_decay_rate=reward_decay_rate,
    contribution_cap=contribution_cap,
    initial_premium_adoption=initial_premium_adoption,
    inactivity_rate=inactivity_rate,
    months=months,
    base_users=base_users,
    customers_per_user=customers_per_user,
    new_customers_per_user=new_customers_per_user,
    initial_token_price=initial_token_price,
    price_elasticity=price_elasticity,
    burn_rate=burn_rate,
    initial_market_sentiment=initial_market_sentiment,
    market_volatility=market_volatility,
    market_trend=market_trend,
    staking_apr=staking_apr,
    reward_pool_size=reward_pool_size,
    num_competitors=num_competitors,
    competitor_growth_rates=competitor_growth_rates,
    competitor_attractiveness=competitor_attractiveness,
    transaction_fee_rate=transaction_fee_rate,
    token_purchase_threshold=token_purchase_threshold,
    token_purchase_amount=token_purchase_amount,
    token_sale_price=token_sale_price,
    total_users_target=total_users_target,
    total_addressable_market=total_addressable_market,
    logistic_enabled=True,
    carrying_capacity=total_addressable_market,
    growth_steepness=0.25,
    midpoint_month=12,
    total_vested_tokens=100_000,
    vest_duration=12,
    shock_events={
        6: {"growth_rate_boost": 0.05, "sentiment_shift": -0.2},
        12: {"growth_rate_boost": -0.03, "inactivity_spike": 0.10},
    }
)

# --- Create a container for the floating charts ---
chart_container = st.container()

# Add custom CSS for modern styling and full-screen modal
st.markdown("""
    <style>
    /* Modern card styling */
    div.stContainer {
        background: linear-gradient(135deg, #f0f2f6, #e0e5ec);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Chart container styling */
    .chart-container {
        background: linear-gradient(135deg, #ffffff, #f0f2f6);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1);
        transition: box-shadow 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 12px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Button styling */
    .stButton>button {
        color: #ffffff !important;
        background-color: #1f77b4 !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #135c8d !important;
    }
    
    /* Improve text readability */
    .chart-title {
        color: #111;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Ensure axis labels are visible */
    .matplotlib-figure text {
        fill: #111 !important;
        font-size: 10pt !important;
    }
    
    /* Ensure the plots fill their containers */
    .stPlot {
        width: 100% !important;
    }
    
    /* Fullscreen modal styling */
    .fullscreen-modal {
        display: none; /* Hide by default */
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.8);
        z-index: 1000;
        justify-content: center;
        align-items: center;
    }

    .modal-content {
        background-color: #fff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        position: relative;
    }

    .close-button {
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 24px;
        color: #333;
        cursor: pointer;
    }

    .close-button:hover {
        color: #000;
    }
    </style>
    
    <script>
    function openFullscreen(chartId) {
        const modal = document.getElementById('modal-' + chartId);
        modal.style.display = 'flex';
    }
    
    function closeFullscreen(chartId) {
        const modal = document.getElementById('modal-' + chartId);
        modal.style.display = 'none';
    }
    </script>
    """, unsafe_allow_html=True)

with chart_container:
    # Header section with key metrics
    st.markdown("### 📊 Token Performance Dashboard")
    
    # Calculate metrics
    token_metrics = get_token_metrics(results)
    user_metrics = get_user_metrics(results)
    health_status, reserve_ratio = calculate_reserve_health(
        results['Reward Pool'].iloc[-1],
        reward_pool_size
    )
    
    # Top row - Token Performance
    st.markdown("#### Token Performance")
    token_cols = st.columns(4)
    with token_cols[0]:
        st.metric(
            "Current Price",
            f"${token_metrics['current_price']:.2f}",
            f"{token_metrics['price_change']:.1f}%"
        )
    with token_cols[1]:
        st.metric("All-Time High", f"${token_metrics['max_price']:.2f}")
    with token_cols[2]:
        st.metric("All-Time Low", f"${token_metrics['min_price']:.2f}")
    with token_cols[3]:
        st.metric(
            "Price Volatility",
            f"{((token_metrics['max_price'] - token_metrics['min_price']) / token_metrics['min_price'] * 100):.1f}%"
        )

    # Middle row - User Growth
    st.markdown("#### User Growth")
    user_cols = st.columns(4)
    with user_cols[0]:
        st.metric(
            "Current Users",
            format_number(user_metrics['current_users']),
            f"{user_metrics['user_growth']:.1f}%"
        )
    with user_cols[1]:
        st.metric("Initial Users", format_number(user_metrics['initial_users']))
    with user_cols[2]:
        st.metric("Peak Users", format_number(user_metrics['peak_users']))
    with user_cols[3]:
        st.metric(
            "Market Penetration",
            f"{(user_metrics['current_users'] / total_addressable_market * 100):.1f}%"
        )

    # Enhanced Reserve Health Section
    st.markdown("#### Reserve Health")
    
    # Calculate health metrics
    status_text, ratio = calculate_reserve_health(
        results['Reward Pool'].iloc[-1],
        reward_pool_size
    )
    
    burn_metrics = calculate_monthly_burn_metrics(
        results['Reward Pool'].iloc[-1],
        reward_pool_size,
        months
    )
    
    # Display reserve metrics with enhanced visual feedback
    reserve_cols = st.columns(4)
    
    with reserve_cols[0]:
        st.metric(
            "Reserve Status",
            status_text,
            f"{(ratio * 100):.1f}% of initial"
        )
    
    with reserve_cols[1]:
        current_reserve_formatted = format_number(results['Reward Pool'].iloc[-1])
        st.metric(
            "Current Reserve",
            current_reserve_formatted,
            help="Current balance in the reward pool"
        )
    
    with reserve_cols[2]:
        initial_reserve_formatted = format_number(reward_pool_size)
        st.metric(
            "Initial Reserve",
            initial_reserve_formatted,
            help="Initial reward pool size"
        )
    
    with reserve_cols[3]:
        monthly_burn_formatted = format_number(burn_metrics['monthly_burn'])
        st.metric(
            "Monthly Burn Rate",
            monthly_burn_formatted,
            f"{burn_metrics['burn_rate_pct']:.1f}% of initial",
            help="Average monthly reduction in reserve balance"
        )
    
    # Add detailed burn analysis if in warning or critical state
    if status_text != "Healthy 🟢":
        st.markdown("---")
        st.markdown("#### Reserve Analysis")
        
        analysis_cols = st.columns(2)
        
        with analysis_cols[0]:
            # Display months until depletion
            months_remaining = burn_metrics['months_remaining']
            if months_remaining != float('inf'):
                st.warning(f"At current burn rate, reserves will be depleted in "
                          f"approximately {months_remaining:.1f} months")
            else:
                st.success("Current burn rate is sustainable")
        
        with analysis_cols[1]:
            # Display recommended actions
            if status_text == "Warning 🟡":
                st.warning("""
                    **Recommended Actions:**
                    - Review reward rates
                    - Analyze token utility
                    - Consider revenue optimization
                """)
            elif status_text == "Critical 🔴":
                st.error("""
                    **Urgent Actions Required:**
                    - Reduce reward rates
                    - Implement emergency measures
                    - Review tokenomics parameters
                """)
    
    # Add reserve trend visualization
    st.markdown("#### Reserve Trend")
    reserve_trend = create_plotly_chart(
        results,
        x="Month",
        y="Reward Pool",
        title="Reserve Balance Over Time",
        y_label="Tokens"
    )
    
    # Add threshold lines
    reserve_trend.add_hline(
        y=reward_pool_size * 0.5,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Warning Threshold (50%)"
    )
    reserve_trend.add_hline(
        y=reward_pool_size * 0.25,
        line_dash="dash",
        line_color="red",
        annotation_text="Critical Threshold (25%)"
    )
    
    st.plotly_chart(reserve_trend, use_container_width=True)

    # Then continue with the "Detailed Analysis" section
    st.markdown("### 📈 Detailed Analysis")
    tabs = st.tabs(["Token Supply", "User Segments", "Market Metrics", "Raw Data"])

    with tabs[0]:
        # Token Supply chart
        token_supply_chart = create_plotly_chart(
            results,
            x="Month",
            y=["Tokens Earned", "Tokens Spent", "Tokens Burned"],
            title="Token Supply Metrics",
            y_label="Tokens"
        )
        st.plotly_chart(token_supply_chart, use_container_width=True)

    with tabs[1]:
        # User Segments chart
        user_segments_chart = create_plotly_chart(
            results,
            x="Month",
            y=["Power Users", "Regular Users", "Casual Users"],
            title="User Segments Over Time",
            y_label="Number of Users"
        )
        st.plotly_chart(user_segments_chart, use_container_width=True)

    with tabs[2]:
        # Market Metrics chart
        market_metrics_chart = create_plotly_chart(
            results,
            x="Month",
            y="Market Sentiment",
            title="Market Sentiment Over Time",
            y_label="Sentiment Score"
        )
        st.plotly_chart(market_metrics_chart, use_container_width=True)

    with tabs[3]:
        # Convert large numbers to float to prevent integer overflow
        display_df = results.copy()
        for column in display_df.select_dtypes(include=['int64', 'float64']):
            display_df[column] = display_df[column].astype('float64')
        
        # Find the line where the dataframe is being displayed and modify it:
        # Instead of:
        # st.dataframe(display_df.style.highlight_max(axis=0))

        # Use this:
        def highlight_numeric_max(df):
            """
            Highlight maximum values only in numeric columns
            """
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if not numeric_cols.empty:
                return df.style.highlight_max(axis=0, subset=numeric_cols)
            return df.style

        # Display the dataframe with highlighting only on numeric columns
        st.dataframe(highlight_numeric_max(display_df))

    # Calculate comprehensive reserve metrics
    reserve_metrics = calculate_reserve_metrics(results, reward_pool_size)
    
    st.markdown("---")
    st.markdown("### 📊 Reserve Health Metrics")
    
    # Display time in states
    st.markdown("#### Time in Critical States")
    state_cols = st.columns(3)
    
    with state_cols[0]:
        st.metric(
            "Warning State",
            f"{reserve_metrics['months_in_warning']} months",
            f"{reserve_metrics['warning_percentage']:.1f}% of time",
            help="Time spent between 25-50% of initial reserve"
        )
    
    with state_cols[1]:
        st.metric(
            "Critical State",
            f"{reserve_metrics['months_in_critical']} months",
            f"{reserve_metrics['critical_percentage']:.1f}% of time",
            help="Time spent below 25% of initial reserve"
        )
    
    with state_cols[2]:
        st.metric(
            "Deficit State",
            f"{reserve_metrics['months_in_deficit']} months",
            f"{reserve_metrics['deficit_percentage']:.1f}% of time",
            help="Time spent with negative reserves"
        )
    
    # Display deficit metrics if any exist
    if reserve_metrics['months_in_deficit'] > 0:
        st.markdown("#### Deficit Analysis")
        deficit_cols = st.columns(2)
        
        with deficit_cols[0]:
            st.metric(
                "Cumulative Deficit",
                format_number(reserve_metrics['cumulative_deficit']),
                help="Total sum of negative reserves across all months"
            )
        
        with deficit_cols[1]:
            st.metric(
                "Average Monthly Deficit",
                format_number(
                    reserve_metrics['cumulative_deficit'] / 
                    reserve_metrics['months_in_deficit']
                ) if reserve_metrics['months_in_deficit'] > 0 else "N/A",
                help="Average deficit amount during negative months"
            )
    
    # Display stability metrics
    st.markdown("#### Reserve Stability")
    stability_cols = st.columns(3)
    
    with stability_cols[0]:
        st.metric(
            "Reserve Volatility",
            f"{reserve_metrics['reserve_volatility']:.1f}%",
            help="Standard deviation of reserve levels relative to initial reserve"
        )
    
    with stability_cols[1]:
        st.metric(
            "Recovery Count",
            str(reserve_metrics['recovery_count']),
            help="Number of times reserves recovered from critical to healthy levels"
        )
    
    with stability_cols[2]:
        st.metric(
            "Minimum Reserve Ratio",
            f"{(reserve_metrics['min_reserve_ratio'] * 100):.1f}%",
            help="Lowest reserve level as percentage of initial reserve"
        )
    
    # Add reserve state timeline
    st.markdown("#### Reserve State Timeline")
    
    # Create state timeline data
    timeline_data = results.copy()
    timeline_data['State'] = 'Healthy'
    timeline_data.loc[
        (timeline_data['Reward Pool'] <= reserve_metrics['warning_threshold']) & 
        (timeline_data['Reward Pool'] > reserve_metrics['critical_threshold']),
        'State'
    ] = 'Warning'
    timeline_data.loc[
        (timeline_data['Reward Pool'] <= reserve_metrics['critical_threshold']) & 
        (timeline_data['Reward Pool'] > 0),
        'State'
    ] = 'Critical'
    timeline_data.loc[timeline_data['Reward Pool'] <= 0, 'State'] = 'Deficit'
    
    # Create timeline chart
    timeline_chart = px.line(
        timeline_data,
        x='Month',
        y='Reward Pool',
        color='State',
        title='Reserve Levels and States Over Time',
        color_discrete_map={
            'Healthy': '#00CC96',
            'Warning': '#FFA15A',
            'Critical': '#EF553B',
            'Deficit': '#AB63FA'
        }
    )
    
    # Add threshold lines
    timeline_chart.add_hline(
        y=reserve_metrics['warning_threshold'],
        line_dash="dash",
        line_color="yellow",
        annotation_text="Warning Threshold"
    )
    timeline_chart.add_hline(
        y=reserve_metrics['critical_threshold'],
        line_dash="dash",
        line_color="red",
        annotation_text="Critical Threshold"
    )
    timeline_chart.add_hline(
        y=0,
        line_dash="dash",
        line_color="purple",
        annotation_text="Deficit Threshold"
    )
    
    st.plotly_chart(timeline_chart, use_container_width=True)

# Add fullscreen modal HTML
for chart_id in ['price', 'revenue', 'supply', 'users']:
    st.markdown(f"""
        <div id="modal-{chart_id}" class="fullscreen-modal">
            <div class="modal-content">
                <span class="close-button" onclick="closeFullscreen('{chart_id}')">&times;</span>
                <div id="fullscreen-chart-{chart_id}"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def analyze_reserve_depletion(results, initial_reserve):
    """
    Analyze why reserves became negative and provide recommendations.
    
    Args:
        results: DataFrame with simulation results
        initial_reserve: Initial reward pool size
    
    Returns:
        dict: Analysis results including causes and recommendations
    """
    # Find when reserves went negative
    depletion_month = None
    for idx, row in results.iterrows():
        if row['Reward Pool'] <= 0:
            depletion_month = row['Month']
            break
    
    if depletion_month is None:
        return None
    
    # Calculate key metrics leading to depletion
    pre_depletion = results[results['Month'] < depletion_month].tail(3)
    
    # Analyze burn rate trend
    avg_monthly_burn = (initial_reserve - pre_depletion['Reward Pool'].iloc[-1]) / (depletion_month - 1)
    burn_rate_pct = (avg_monthly_burn / initial_reserve) * 100
    
    # Analyze revenue vs rewards
    avg_monthly_revenue = pre_depletion['Platform Revenue ($)'].mean()
    avg_monthly_rewards = pre_depletion['Reward'].mean() * pre_depletion['Users'].mean()
    
    # Identify primary causes
    causes = []
    recommendations = []
    
    if burn_rate_pct > 10:
        causes.append(f"High burn rate ({burn_rate_pct:.1f}% monthly)")
        recommendations.append("Reduce reward rates to sustainable levels")
    
    if avg_monthly_rewards > avg_monthly_revenue * 2:
        causes.append("Rewards significantly exceed platform revenue")
        recommendations.append("Increase transaction fees or implement additional revenue streams")
    
    user_growth = (pre_depletion['Users'].iloc[-1] / pre_depletion['Users'].iloc[0] - 1) * 100
    if user_growth > 50:
        causes.append(f"Rapid user growth ({user_growth:.1f}% over 3 months)")
        recommendations.append("Implement progressive reward reduction with user growth")
    
    if pre_depletion['Market Sentiment'].mean() < 0.8:
        causes.append("Low market sentiment affecting token utility")
        recommendations.append("Enhance token utility and implement buy-back mechanisms")
    
    return {
        'depletion_month': depletion_month,
        'avg_monthly_burn': avg_monthly_burn,
        'burn_rate_pct': burn_rate_pct,
        'avg_monthly_revenue': avg_monthly_revenue,
        'avg_monthly_rewards': avg_monthly_rewards,
        'causes': causes,
        'recommendations': recommendations
    }

def run_sensitivity_analysis(base_params, param_ranges):
    results = {}
    
    for param_name, param_range in param_ranges.items():
        param_results = []
        base_value = base_params[param_name]
        
        for variation in param_range:
            # Create modified parameters
            test_params = base_params.copy()
            test_params[param_name] = variation
            
            # Run simulation with modified parameters
            sim_results = simulate_tokenomics(**test_params)
            
            # Calculate key metrics
            final_reserve = sim_results['Reward Pool'].iloc[-1]
            reserve_ratio = final_reserve / base_params['reward_pool_size']
            
            param_results.append({
                'variation': variation,
                'final_reserve': final_reserve,
                'reserve_ratio': reserve_ratio,
                'percent_change': ((variation - base_value) / base_value) * 100
            })
        
        results[param_name] = param_results
    
    return results

def create_sensitivity_chart(sensitivity_results, param_name):
    """Create a chart showing parameter sensitivity"""
    data = pd.DataFrame(sensitivity_results[param_name])
    
    fig = px.line(
        data,
        x='percent_change',
        y='reserve_ratio',
        title=f'Reserve Sensitivity to {param_name}',
        labels={
            'percent_change': 'Parameter Change (%)',
            'reserve_ratio': 'Final Reserve Ratio'
        },
        template="plotly_dark"
    )
    
    # Add reference line for base case
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Base Case"
    )
    
    # Update layout
    fig.update_layout(
        hovermode='x unified',
        showlegend=False
    )
    
    return fig

# Add to your UI section:
with st.sidebar.expander("Sensitivity Analysis"):
    run_sensitivity = st.checkbox("Run Sensitivity Analysis", value=False)
    
    if run_sensitivity:
        st.markdown("### Parameter Ranges for Analysis")
        
        # Define parameter ranges to test
        param_ranges = {
            'burn_rate': st.slider(
                "Burn Rate Range",
                min_value=0.0,
                max_value=0.10,
                value=(0.01, 0.05),
                step=0.01
            ),
            'initial_token_price': st.slider(
                "Token Price Range",
                min_value=0.1,
                max_value=5.0,
                value=(0.5, 2.0),
                step=0.1
            ),
            'reward_decay_rate': st.slider(
                "Reward Decay Range",
                min_value=0.0,
                max_value=0.10,
                value=(0.01, 0.05),
                step=0.01
            ),
            'transaction_fee_rate': st.slider(
                "Transaction Fee Range",
                min_value=0.0,
                max_value=0.10,
                value=(0.02, 0.06),
                step=0.01
            )
        }

# In your main UI section, after the reserve health metrics:
if run_sensitivity:
    st.markdown("---")
    st.markdown("### 📊 Sensitivity Analysis")
    
    # Prepare base parameters
    base_params = {
        'initial_reward': 0.35,
        'initial_search_fee': 5.0,
        'growth_rate': 0.06,
        'line_items_per_customer': 75,
        'initial_lookup_frequency': 9,
        'reward_decay_rate': 1.5,
        'contribution_cap': 750,
        'initial_premium_adoption': 0.25,
        'inactivity_rate': 0.04,
        'months': 36,
        'base_users': 1000,
        'customers_per_user': 50,
        'new_customers_per_user': 7,
        'initial_token_price': 1.0,
        'price_elasticity': 0.4,
        'burn_rate': 0.04,
        'initial_market_sentiment': 1.0,
        'market_volatility': 0.15,
        'market_trend': 0.0,
        'staking_apr': 7.5,
        'reward_pool_size': 1000000,
        'num_competitors': 3,
        'competitor_growth_rates': [0.04, 0.04, 0.04],
        'competitor_attractiveness': [0.02, 0.02, 0.02],
        'transaction_fee_rate': 0.04,
        'token_purchase_threshold': 7.5,
        'token_purchase_amount': 15.0,
        'token_sale_price': 1.0,
        'total_users_target': 100000,
        'total_addressable_market': 1000000,
        'logistic_enabled': True,
        'carrying_capacity': 1000000,
        'growth_steepness': 0.25,
        'midpoint_month': 12,
        'total_vested_tokens': 100000,
        'vest_duration': 12,
        'shock_events': {
            6: {"growth_rate_boost": 0.05, "sentiment_shift": -0.2},
            12: {"growth_rate_boost": -0.03, "inactivity_spike": 0.10},
        }
    }
    
    # Generate test ranges
    test_ranges = {
        param: np.linspace(range_vals[0], range_vals[1], 5)
        for param, range_vals in param_ranges.items()
    }
    
    # Run sensitivity analysis
    sensitivity_results = run_sensitivity_analysis(base_params, test_ranges)
    
    # Display results
    st.markdown("#### Parameter Impact on Reserves")
    
    # Create tabs for different parameters
    param_tabs = st.tabs(list(test_ranges.keys()))
    
    for tab, param_name in zip(param_tabs, test_ranges.keys()):
        with tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display sensitivity chart
                chart = create_sensitivity_chart(sensitivity_results, param_name)
                st.plotly_chart(chart, use_container_width=True)
            
            with col2:
                # Display key findings
                results = sensitivity_results[param_name]
                
                # Find most impactful change
                max_impact = max(results, key=lambda x: abs(1 - x['reserve_ratio']))
                
                st.markdown("#### Key Findings")
                st.markdown(f"""
                    - Base value: {base_params[param_name]:.3f}
                    - Most impactful change: {max_impact['variation']:.3f}
                    - Impact on reserves: {((max_impact['reserve_ratio'] - 1) * 100):.1f}%
                """)
                
                # Add recommendations
                st.markdown("#### Recommendations")
                if max_impact['reserve_ratio'] > 1.1:
                    st.success(f"Consider increasing {param_name} to improve reserve health")
                elif max_impact['reserve_ratio'] < 0.9:
                    st.warning(f"Current {param_name} may be too high for sustainable reserves")
                else:
                    st.info(f"Current {param_name} is within a stable range")
    
    # Add overall sensitivity summary
    st.markdown("#### Overall Sensitivity Summary")
    
    # Calculate sensitivity scores
    sensitivity_scores = {}
    for param_name, results in sensitivity_results.items():
        variations = [r['reserve_ratio'] for r in results]
        sensitivity_scores[param_name] = np.std(variations)
    
    # Display parameters sorted by sensitivity
    sorted_params = sorted(
        sensitivity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for param_name, score in sorted_params:
        st.markdown(f"""
            - **{param_name}**: Sensitivity Score = {score:.3f}
            ({'+/-' if score > 0.1 else '~'}{score * 100:.1f}% reserve impact)
        """)

def analyze_shock_impact(results, shock_events):
    """
    Analyze how shock events affect reserve depletion.
    
    Args:
        results: DataFrame with simulation results
        shock_events: Dict of shock events by month
    
    Returns:
        dict: Analysis of shock impacts
    """
    if not shock_events:
        return None
    
    impacts = []
    for month, event in shock_events.items():
        # Get data before and after shock
        pre_shock = results[results['Month'] < month].tail(3)
        post_shock = results[results['Month'] >= month].head(3)
        
        # Calculate immediate impact
        pre_reserve = pre_shock['Reward Pool'].iloc[-1]
        post_reserve = post_shock['Reward Pool'].iloc[0]
        reserve_change = ((post_reserve / pre_reserve) - 1) * 100
        
        # Calculate burn rate changes
        pre_burn = (pre_shock['Reward Pool'].iloc[0] - pre_shock['Reward Pool'].iloc[-1]) / 3
        post_burn = (post_shock['Reward Pool'].iloc[0] - post_shock['Reward Pool'].iloc[-1]) / 3
        burn_rate_change = ((post_burn / pre_burn) - 1) * 100 if pre_burn != 0 else 0
        
        # Collect event details
        event_details = []
        if "growth_rate_boost" in event:
            event_details.append(f"Growth {'+' if event['growth_rate_boost'] > 0 else ''}{event['growth_rate_boost']*100:.1f}%")
        if "sentiment_shift" in event:
            event_details.append(f"Sentiment {'+' if event['sentiment_shift'] > 0 else ''}{event['sentiment_shift']:.2f}")
        if "inactivity_spike" in event:
            event_details.append(f"Churn +{event['inactivity_spike']*100:.1f}%")
        if "price_shock" in event:
            event_details.append(f"Price x{event['price_shock']:.2f}")
        if "reward_pool_change" in event:
            event_details.append(f"Rewards x{event['reward_pool_change']:.2f}")
        
        impacts.append({
            'month': month,
            'event_description': " | ".join(event_details),
            'reserve_change': reserve_change,
            'burn_rate_change': burn_rate_change,
            'pre_reserve': pre_reserve,
            'post_reserve': post_reserve,
            'pre_burn': pre_burn,
            'post_burn': post_burn
        })
    
    return impacts

# Update the UI section where shock events are displayed:
if shock_events:
    st.markdown("---")
    st.markdown("### 🌊 Shock Event Impact Analysis")
    
    shock_impacts = analyze_shock_impact(results, shock_events)
    
    if shock_impacts:
        # Create impact timeline
        impact_data = []
        for impact in shock_impacts:
            impact_data.append({
                'Month': impact['month'],
                'Event': impact['event_description'],
                'Reserve Change': f"{impact['reserve_change']:.1f}%",
                'Burn Rate Change': f"{impact['burn_rate_change']:.1f}%"
            })
        
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(
            impact_df,
            column_config={
                'Month': st.column_config.NumberColumn(
                    'Month',
                    help='Month when shock occurred'
                ),
                'Event': st.column_config.TextColumn(
                    'Event Description',
                    help='Type and magnitude of shock'
                ),
                'Reserve Change': st.column_config.TextColumn(
                    'Reserve Impact',
                    help='Immediate change in reserve level'
                ),
                'Burn Rate Change': st.column_config.TextColumn(
                    'Burn Rate Impact',
                    help='Change in monthly burn rate'
                )
            },
            hide_index=True
        )
        
        # Display cumulative impact
        st.markdown("#### Cumulative Impact Analysis")
        impact_cols = st.columns(3)
        
        total_reserve_change = sum(impact['reserve_change'] for impact in shock_impacts)
        avg_burn_rate_change = np.mean([impact['burn_rate_change'] for impact in shock_impacts])
        
        with impact_cols[0]:
            st.metric(
                "Total Reserve Impact",
                f"{total_reserve_change:.1f}%",
                help="Cumulative change in reserves from all shocks"
            )
        
        with impact_cols[1]:
            st.metric(
                "Average Burn Rate Impact",
                f"{avg_burn_rate_change:.1f}%",
                help="Average change in burn rate after shocks"
            )
        
        with impact_cols[2]:
            critical_shocks = sum(1 for impact in shock_impacts if impact['reserve_change'] < -10)
            st.metric(
                "Critical Shocks",
                str(critical_shocks),
                help="Number of shocks causing >10% reserve drop"
            )
        
        # Display shock event timeline
        st.markdown("#### Shock Event Timeline")
        
        # Create timeline chart
        timeline_fig = px.line(
            results,
            x='Month',
            y='Reward Pool',
            title='Reserve Levels with Shock Events',
            template="plotly_dark"
        )
        
        # Add vertical lines for shock events
        for impact in shock_impacts:
            timeline_fig.add_vline(
                x=impact['month'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Shock: {impact['event_description']}",
                annotation_position="top right"
            )
        
        # Add threshold lines
        timeline_fig.add_hline(
            y=results['Reward Pool'].iloc[0] * 0.5,
            line_dash="dash",
            line_color="yellow",
            annotation_text="Warning Threshold"
        )
        timeline_fig.add_hline(
            y=results['Reward Pool'].iloc[0] * 0.25,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold"
        )
        
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Add recovery analysis
        st.markdown("#### Recovery Analysis")
        recovery_cols = st.columns(2)
        
        with recovery_cols[0]:
            # Calculate average recovery time
            recovery_times = []
            for impact in shock_impacts:
                post_shock = results[results['Month'] >= impact['month']]
                recovery_month = None
                for idx, row in post_shock.iterrows():
                    if row['Reward Pool'] >= impact['pre_reserve']:
                        recovery_month = row['Month']
                        break
                if recovery_month:
                    recovery_times.append(recovery_month - impact['month'])
            
            avg_recovery = np.mean(recovery_times) if recovery_times else None
            if avg_recovery:
                st.metric(
                    "Average Recovery Time",
                    f"{avg_recovery:.1f} months",
                    help="Average time to return to pre-shock levels"
                )
            else:
                st.warning("No full recoveries observed")
        
        with recovery_cols[1]:
            # Calculate permanent impact
            final_impact = ((results['Reward Pool'].iloc[-1] / results['Reward Pool'].iloc[0]) - 1) * 100
            st.metric(
                "Permanent Impact",
                f"{final_impact:.1f}%",
                help="Long-term change in reserves from initial level"
            )

def simulate_user_growth(users, growth_rate, max_users, logistic_enabled=False, carrying_capacity=None):
    """
    Simulate user growth with optional logistic growth model.
    
    Args:
        users: Current number of users
        growth_rate: Monthly growth rate
        max_users: Maximum number of users
        logistic_enabled: Whether to use logistic growth
        carrying_capacity: Carrying capacity for logistic growth
    
    Returns:
        int: Updated number of users
    """
    if logistic_enabled and carrying_capacity:
        # Logistic growth model
        growth_factor = growth_rate * (1 - users / carrying_capacity)
        new_users = users + int(users * growth_factor)
    else:
        # Linear growth model
        new_users = users + int(users * growth_rate)
    
    return min(new_users, max_users)

def create_enhanced_plotly_chart(df, x, y, title, y_label):
    fig = px.line(
        df,
        x=x,
        y=y,
        title=title,
        labels={x: x, y: y_label},
        template="plotly_dark",
        width=800,
        height=450
    )
    
    # Add annotations for significant events
    fig.add_annotation(
        x=10,  # Example month
        y=df[y].max(),
        text="Peak Value",
        showarrow=True,
        arrowhead=1
    )
    
    # Update layout for better readability
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            font=dict(size=20),
            xanchor='center',
            yanchor='top'
        ),
        margin=dict(l=60, r=30, t=50, b=60),
        yaxis_title=y_label,
        xaxis_title="Month",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        hovermode='x unified'
    )
    
    return fig# Ensuring Git detects changes
