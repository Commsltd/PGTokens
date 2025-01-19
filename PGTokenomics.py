import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from tokenomics_optimizer import run_tokenomics_optimization
from simulation import simulate_tokenomics

# Must be the first Streamlit command
st.set_page_config(
    page_title="PG Tokenomics Simulator",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for debug logs
if 'debug_logs' not in st.session_state:
    st.session_state['debug_logs'] = []

# Centralized logging function
def log_message(message, debug_only=True):
    """
    Logs a message to the appropriate section based on the debug_only flag.
    
    Args:
        message (str): The message to log.
        debug_only (bool): If True, log only in the debugging section.
    """
    if debug_only:
        # Store messages for debugging section
        st.session_state['debug_logs'].append(message)
    else:
        # Display messages in the main UI
        st.write(message)

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
    """Calculate comprehensive reserve metrics with dynamic thresholds."""
    # Calculate dynamic thresholds based on user count and activity
    def calculate_dynamic_thresholds(row, initial_users, initial_reserve):
        """Calculate warning and critical thresholds based on current metrics."""
        user_factor = row['Users'] / initial_users
        activity_factor = (row['Tokens Spent'] + row['Premium Users'] * 10) / initial_reserve
        
        # Thresholds become stricter as user base and activity grow
        warning_threshold = initial_reserve * 0.5 * (1 + np.log1p(user_factor * activity_factor))
        critical_threshold = initial_reserve * 0.25 * (1 + np.log1p(user_factor * activity_factor))
        
        return warning_threshold, critical_threshold
    
    # Get initial users
    initial_users = results['Users'].iloc[0]
    
    # Calculate dynamic thresholds for each month
    thresholds = pd.DataFrame({
        'warning_threshold': results.apply(
            lambda x: calculate_dynamic_thresholds(x, initial_users, initial_reserve)[0], 
            axis=1
        ),
        'critical_threshold': results.apply(
            lambda x: calculate_dynamic_thresholds(x, initial_users, initial_reserve)[1], 
            axis=1
        )
    })
    
    # Initialize state tracking with dynamic thresholds
    states = {
        "healthy": lambda x, w: x > w,
        "warning": lambda x, w, c: c < x <= w,
        "critical": lambda x, c: 0 < x <= c,
        "deficit": lambda x: x <= 0
    }
    
    # Track state transitions and recoveries
    current_state = "healthy"
    recovery_events = []
    consecutive_months_below = 0
    lowest_point = float('inf')
    state_history = []
    
    for i, row in results.iterrows():
        reserve = row['Reward Pool']
        warning_threshold = thresholds.loc[i, 'warning_threshold']
        critical_threshold = thresholds.loc[i, 'critical_threshold']
        
        # Determine current state with dynamic thresholds
        if states["healthy"](reserve, warning_threshold):
            new_state = "healthy"
        elif states["warning"](reserve, warning_threshold, critical_threshold):
            new_state = "warning"
        elif states["critical"](reserve, critical_threshold):
            new_state = "critical"
        else:
            new_state = "deficit"
        
        state_history.append(new_state)
        
        # Track consecutive months below warning threshold
        if new_state in ["warning", "critical", "deficit"]:
            consecutive_months_below += 1
            lowest_point = min(lowest_point, reserve)
        else:
            # Record recovery if returning to healthy from a non-healthy state
            if consecutive_months_below >= 2 and current_state != "healthy":
                recovery_events.append({
                    'month': i,
                    'from_state': current_state,
                    'duration': consecutive_months_below,
                    'lowest_point': lowest_point,
                    'recovery_magnitude': (reserve - lowest_point) / initial_reserve
                })
            consecutive_months_below = 0
            lowest_point = float('inf')
        
        current_state = new_state
    
    # Calculate time in each state
    total_months = len(results)
    state_counts = pd.Series(state_history).value_counts()
    
    metrics = {
        'months_in_warning': state_counts.get('warning', 0),
        'months_in_critical': state_counts.get('critical', 0),
        'months_in_deficit': state_counts.get('deficit', 0),
        'warning_percentage': (state_counts.get('warning', 0) / total_months) * 100,
        'critical_percentage': (state_counts.get('critical', 0) / total_months) * 100,
        'deficit_percentage': (state_counts.get('deficit', 0) / total_months) * 100,
        'min_reserve_ratio': results['Reward Pool'].min() / initial_reserve,  # Add this line
        'warning_threshold': thresholds['warning_threshold'].mean(),
        'critical_threshold': thresholds['critical_threshold'].mean()
    }
    
    # Calculate recovery strength with weighted impact
    recovery_strength = 0
    if recovery_events:
        weighted_recoveries = [
            event['recovery_magnitude'] * (1 + 0.5 * (event['from_state'] == 'critical'))
            for event in recovery_events
        ]
        recovery_strength = np.mean(weighted_recoveries)
    
    # Add threshold and state history to results
    metrics.update({
        'dynamic_thresholds': thresholds,
        'state_history': state_history,
        'recovery_events': recovery_events,
        'recovery_count': len(recovery_events),
        'recovery_strength': recovery_strength * 100,
        'avg_warning_threshold': thresholds['warning_threshold'].mean(),
        'avg_critical_threshold': thresholds['critical_threshold'].mean(),
        'current_state': current_state,
        'consecutive_months_below': consecutive_months_below
    })
    
    # Calculate stability score based on state percentages
    stability_score = 100 * (1 - (
        metrics['warning_percentage'] + 
        metrics['critical_percentage'] * 2 + 
        metrics['deficit_percentage'] * 3
    ) / 300)  # Weighted impact of different states
    
    # Add stability score to metrics dictionary
    metrics.update({
        'stability_score': stability_score  # Add this line
    })
    
    return metrics

def calculate_monthly_burn_metrics(current_reserve, initial_reserve, months, results_df):
    """Calculate comprehensive burn rate metrics accounting for rewards, revenue, and token burning."""
    # Calculate recent reserve changes (last 3 months)
    recent_months = min(3, months)
    recent_reserve_change = results_df['Reward Pool'].iloc[-recent_months:].diff().fillna(0)
    
    # Separate growth from burn
    recent_burns = recent_reserve_change[recent_reserve_change < 0]
    recent_growth = recent_reserve_change[recent_reserve_change > 0]
    
    # Calculate average monthly burn (only from periods with actual burns)
    recent_monthly_burn = abs(recent_burns.mean()) if not recent_burns.empty else 0
    recent_monthly_growth = recent_growth.mean() if not recent_growth.empty else 0
    
    # Calculate net burn/growth rate
    net_monthly_change = recent_reserve_change.mean()
    
    # Calculate burn rate relative to revenue (using absolute values)
    recent_revenue = results_df['Platform Revenue ($)'].iloc[-recent_months:].mean()
    revenue_burn_ratio = recent_monthly_burn / recent_revenue if recent_revenue > 0 else 0
    
    # Calculate sustainable burn rate (cap at 50% of revenue or 5% of current reserve, whichever is lower)
    max_sustainable_from_revenue = recent_revenue * 0.5
    max_sustainable_from_reserve = current_reserve * 0.05
    sustainable_burn = min(max_sustainable_from_revenue, max_sustainable_from_reserve)
    
    # Determine if current burn rate is sustainable
    is_sustainable = (recent_monthly_burn <= sustainable_burn) if recent_monthly_burn > 0 else True
    
    # Calculate months until depletion (only if there's net negative change)
    if net_monthly_change < 0:
        months_remaining = current_reserve / abs(net_monthly_change)
    else:
        months_remaining = float('inf')
    
    return {
        'monthly_burn': recent_monthly_burn,
        'monthly_growth': recent_monthly_growth,
        'net_monthly_change': net_monthly_change,
        'burn_rate_pct': (recent_monthly_burn / initial_reserve * 100) if recent_monthly_burn > 0 else 0,
        'growth_rate_pct': (recent_monthly_growth / initial_reserve * 100) if recent_monthly_growth > 0 else 0,
        'months_remaining': months_remaining,
        'revenue_burn_ratio': revenue_burn_ratio,
        'is_sustainable': is_sustainable,
        'sustainable_burn': sustainable_burn,
        'burn_sustainability': ((sustainable_burn - recent_monthly_burn) / sustainable_burn * 100) if sustainable_burn > 0 and recent_monthly_burn > 0 else 100
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
        y="Reward Pool",
        title="Platform Revenue & Treasury Tokens Over Time",
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
            text="Platform Revenue & Treasury Tokens Over Time",
            x=0.5,
            y=0.95,
            font=dict(size=20),
            xanchor='center',
            yanchor='top'
        ),
        yaxis=dict(
            title="Treasury Tokens",
            titlefont=dict(color="#636EFA"),  # Blue color for treasury
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
    user_growth_rate,
    transaction_volume,
    staked_ratio,
    current_month,  # Add current_month parameter
    price_history
):
    """Calculate token price with dampened factors and realistic bounds."""
    if total_tokens_issued <= 0:
        return base_price

    # Calculate supply/demand ratio with tighter bounds
    supply_demand_ratio = np.clip(
        total_tokens_spent / total_tokens_issued if total_tokens_issued > 0 else 0,
        0.2,  # Minimum 0.2x impact
        2.0   # Maximum 2x impact
    )
    
    # Calculate velocity dampening with reduced impact
    velocity = transaction_volume / total_tokens_issued if total_tokens_issued > 0 else 1
    velocity_factor = 1 / (1 + velocity * 0.5)  # Reduced velocity impact
    velocity_factor = np.clip(velocity_factor, 0.5, 1.5)  # Bound velocity impact
    
    # Calculate staking impact with diminishing returns
    staking_factor = 1 + (staked_ratio * 0.3)  # Changed from staking_ratio to staked_ratio
    staking_factor = min(1.3, staking_factor)  # Cap at 30% boost
    
    # Calculate growth premium with diminishing returns
    growth_premium = 1 + max(0, min(user_growth_rate * 1.5, 0.5))  # Cap at 50% premium
    
    # Apply sentiment with reduced volatility
    dampened_sentiment = 1 + (market_sentiment - 1) * 0.7  # Reduce sentiment impact
    
    # Calculate base price impact
    price = (
        base_price * 
        (supply_demand_ratio ** (price_elasticity * 0.7)) *  # Reduce elasticity impact
        dampened_sentiment * 
        velocity_factor * 
        staking_factor * 
        growth_premium
    )
    
    # Apply progressive price bounds
    max_price_multiplier = 1 + (np.log1p(current_month) * 0.5)  # Use current_month parameter
    max_price = base_price * max_price_multiplier
    min_price = base_price * 0.2  # Floor at 20% of initial price
    
    # Smooth price changes
    if len(price_history) > 0:
        previous_price = price_history[-1]
        max_change = previous_price * 0.15  # Max 15% change per period
        price = np.clip(
            price,
            previous_price - max_change,
            previous_price + max_change
        )
    
    return np.clip(price, min_price, max_price)

def simulate_market_sentiment(
    initial_sentiment,
    volatility,
    trend,
    user_growth,
    price_change,
    transaction_volume_change,
    sentiment_history=None  # Add sentiment history parameter
):
    """
    Simulates market sentiment changes with smoothing and realistic market psychology.
    """
    # Initialize sentiment history if None
    if sentiment_history is None:
        sentiment_history = []
    
    # Calculate base sentiment change with reduced volatility
    base_change = np.random.normal(0, volatility * 0.7) + (trend * 0.5)
    
    # Apply moving average smoothing if we have history
    if sentiment_history:
        last_changes = np.diff(sentiment_history[-3:]) if len(sentiment_history) >= 3 else [0]
        avg_change = np.mean(last_changes)
        base_change = 0.7 * base_change + 0.3 * avg_change
    
    # Calculate user growth impact with momentum
    growth_impact = 0
    if user_growth > 0:
        growth_impact = min(user_growth * 0.08, 0.03)  # Reduced from 0.1 to 0.08
    else:
        growth_impact = max(user_growth * 0.12, -0.04)  # Stronger negative impact
    
    # Calculate price momentum effect
    price_momentum = 0
    if len(sentiment_history) >= 2:
        price_trend = np.mean([price_change, price_change * 0.5])  # Weighted recent price change
        price_momentum = np.clip(price_trend * 0.08, -0.03, 0.03)
    
    # Calculate volume impact with threshold
    volume_impact = 0
    if abs(transaction_volume_change) > 0.1:  # Only consider significant volume changes
        volume_impact = np.clip(transaction_volume_change * 0.05, -0.02, 0.02)
    
    # Combine all factors with weighted importance
    sentiment_change = (
        base_change * 0.4 +          # Base random walk (40%)
        growth_impact * 0.3 +        # User growth impact (30%)
        price_momentum * 0.2 +       # Price momentum (20%)
        volume_impact * 0.1          # Volume impact (10%)
    )
    
    # Apply mean reversion
    distance_from_neutral = initial_sentiment - 1.0
    mean_reversion = -distance_from_neutral * 0.1  # 10% reversion to mean
    sentiment_change += mean_reversion
    
    # Calculate new sentiment with tighter bounds
    new_sentiment = initial_sentiment + sentiment_change
    new_sentiment = np.clip(new_sentiment, 0.7, 1.3)  # Tighter bounds
    
    return new_sentiment

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

def logistic_growth(current_month, carrying_capacity, initial_base_users, growth_steepness, midpoint):
    """
    Implements an S-curve growth model using the logistic function with market saturation effects.
    
    Args:
        current_month: Current month in simulation
        carrying_capacity: Maximum possible users (TAM)
        initial_base_users: Starting number of users
        growth_steepness: Controls how steep the S-curve is (0.1-0.5)
        midpoint: Month at which growth is ~50% of capacity
    
    Returns:
        int: Projected number of users at this point
    """
    # Ensure parameters are within valid ranges
    growth_steepness = np.clip(growth_steepness, 0.1, 0.5)
    carrying_capacity = max(carrying_capacity, initial_base_users * 2)
    midpoint = max(1, midpoint)
    
    # Calculate market saturation factor (reduces growth as we approach capacity)
    saturation_factor = 1 - (initial_base_users / carrying_capacity)
    
    # Apply saturation-adjusted logistic growth
    growth_factor = -growth_steepness * saturation_factor * (current_month - midpoint)
    user_count = carrying_capacity / (1 + np.exp(growth_factor))
    
    # Apply additional growth dampening near capacity
    proximity_to_capacity = user_count / carrying_capacity
    if proximity_to_capacity > 0.7:  # Start dampening at 70% of capacity
        dampening = 1 - ((proximity_to_capacity - 0.7) / 0.3)  # Linear reduction
        user_count *= dampening
    
    # Ensure we don't go below initial users or above carrying capacity
    return int(max(initial_base_users, min(user_count, carrying_capacity * 0.95)))  # Cap at 95% of TAM

def calculate_competitor_impact(
    users, 
    token_price, 
    initial_token_price, 
    competitor_attractiveness,
    market_saturation
):
    """
    Calculate enhanced competitor impact with market saturation effects.
    
    Args:
        users: Current number of users
        token_price: Current token price
        initial_token_price: Initial token price
        competitor_attractiveness: Base competitor attractiveness
        market_saturation: Current market saturation level
    
    Returns:
        float: Adjusted competitor churn rate
    """
    # Base effect from token price
    price_effect = max(0, 1 - token_price / initial_token_price)
    
    # Market saturation increases competition
    saturation_multiplier = 1 + (market_saturation * 2)  # 2x effect at full saturation
    
    # Calculate final effect with bounds
    competitor_effect = (
        competitor_attractiveness * 
        price_effect * 
        saturation_multiplier
    )
    
    return np.clip(competitor_effect, 0.01, 0.05)  # Bound between 1-5% monthly churn

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
    """Applies shock event modifications with enhanced impact calculations."""
    params = current_params.copy()
    
    # Track cascading effects
    cascade_effects = {
        "sentiment_impact": 0,
        "price_impact": 0,
        "user_impact": 0,
        "revenue_impact": 0
    }
    
    # Growth rate modifications with cascading effects
    if "growth_rate_boost" in event_info:
        base_adjustment = event_info["growth_rate_boost"]
        # Amplify negative shocks significantly more
        if base_adjustment < 0:
            adjustment = base_adjustment * 2.0  # Double negative impact
            cascade_effects["sentiment_impact"] -= 0.3  # Large sentiment hit
            cascade_effects["price_impact"] -= 0.2  # Price pressure
        else:
            adjustment = base_adjustment * 1.3  # 30% stronger positive impact
            cascade_effects["sentiment_impact"] += 0.1
        
        params["growth_rate"] = max(0, min(1.0, params.get("growth_rate", 0) + adjustment))
        params["growth_shock_recovery"] = 6  # Takes 6 months to recover

    # Sentiment modifications with market psychology
    if "sentiment_shift" in event_info:
        base_shift = event_info["sentiment_shift"]
        # Add market psychology effects
        if base_shift < 0:
            cascade_effects["price_impact"] -= 0.15  # Price pressure
            cascade_effects["user_impact"] -= 0.1  # User confidence impact
        params["sentiment_shift"] = base_shift * 1.5  # 50% stronger
        params["sentiment_persistence"] = 4  # Longer persistence
        params["market_sentiment"] = max(0.1, min(2.0, 
            params.get("market_sentiment", 1.0) + base_shift + cascade_effects["sentiment_impact"]))

    # Inactivity spike with network effects
    if "inactivity_spike" in event_info:
        base_spike = event_info["inactivity_spike"] * 1.5  # 50% stronger
        params["inactivity_rate"] = max(0.01, min(0.3, 
            params.get("inactivity_rate", 0.05) + base_spike))
        params["competitor_multiplier"] = 2.0  # Doubled competitor effectiveness
        cascade_effects["sentiment_impact"] -= 0.2
        cascade_effects["revenue_impact"] -= 0.15

    # Price shocks with market contagion
    if "price_shock" in event_info:
        shock_multiplier = event_info["price_shock"]
        params["token_price_multiplier"] = shock_multiplier
        
        # Market contagion effects
        if shock_multiplier < 1:
            cascade_effects["sentiment_impact"] -= 0.25
            cascade_effects["user_impact"] -= 0.15
            params["staking_modifier"] = 0.5  # Reduced staking
        else:
            cascade_effects["sentiment_impact"] += 0.15
            params["staking_modifier"] = 1.5  # Increased staking

    # Reward pool modifications with ecosystem impact
    if "reward_pool_change" in event_info:
        change_multiplier = event_info["reward_pool_change"]
        params["reward_pool_modifier"] = change_multiplier
        
        if change_multiplier < 1:
            cascade_effects["sentiment_impact"] -= 0.2
            cascade_effects["user_impact"] -= 0.1
            params["staking_modifier"] = 0.7
        
        # Add ecosystem effects
        params["reward_adjustment"] = 1 + (change_multiplier - 1) * 0.7

    # Apply cascade effects
    params["cascade_effects"] = cascade_effects
    params["shock_intensity"] = sum(abs(v) for v in cascade_effects.values())

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
    reward_pool_share,
    burn_share,
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
            "contribution_multiplier": 3.0,  # Reduced from 5.0 to 3.0
            "lookup_multiplier": 1.5,        # Reduced from 2.0 to 1.5
            "premium_adoption_multiplier": 1.5,  # Reduced from 2.0 to 1.5
            "churn_resistance": 0.8  # Power users are 20% less likely to churn
        },
        "regular": {
            "proportion": 0.6,
            "contribution_multiplier": 1.0,
            "lookup_multiplier": 1.0,
            "premium_adoption_multiplier": 1.0,
            "churn_resistance": 1.0
        },
        "casual": {
            "proportion": 0.3,
            "contribution_multiplier": 0.3,  # Increased from 0.2 to 0.3
            "lookup_multiplier": 0.5,
            "premium_adoption_multiplier": 0.5,
            "churn_resistance": 1.2  # Casual users are 20% more likely to churn
        }
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

    # Initialize tracking variables
    previous_price = initial_token_price
    previous_transaction_volume = 0
    price_history = []
    
    # Initialize revenue tracking
    fiat_transaction_revenue = 0
    fiat_premium_revenue = 0
    token_transaction_revenue = 0
    token_premium_revenue = 0
    token_search_revenue = 0
    treasury_tokens = 0  # Track platform's treasury tokens
    
    # Initialize enhanced shock tracking
    shock_effects = {
        "sentiment_persistence": 0,
        "growth_shock_recovery": 0,
        "competitor_multiplier": 1.0,
        "staking_modifier": 1.0,
        "cascade_effects": {},
        "shock_intensity": 0,
        "segment_impact": False  # Add this line
    }

    # Add sentiment history tracking
    sentiment_history = []
    price_change_history = []

    # --- Simulation Loop ---
    for month in range(months):
        current_month = month + 1
        
        # Apply shock events if any exist for this month
        if shock_events and current_month in shock_events:
            # Get modified parameters for this month
            modified_params = apply_shock_event(current_params, shock_events[current_month])
            
            # Check if any segment modifiers exist in the event
            has_segment_impact = any(
                key.endswith("_modifier") and key.split("_")[0] in user_segments
                for key in shock_events[current_month].keys()
            )
            shock_effects["segment_impact"] = has_segment_impact
            
            # Apply immediate parameter changes
            effective_growth_rate *= (1 + modified_params["cascade_effects"]["user_impact"])
            market_sentiment = modified_params["market_sentiment"]
            inactivity_rate = modified_params["inactivity_rate"]
            
            # Store shock effects
            shock_effects.update({
                "sentiment_persistence": modified_params.get("sentiment_persistence", 0),
                "growth_shock_recovery": modified_params.get("growth_shock_recovery", 0),
                "competitor_multiplier": modified_params.get("competitor_multiplier", 1.0),
                "staking_modifier": modified_params.get("staking_modifier", 1.0),
                "cascade_effects": modified_params["cascade_effects"],
                "shock_intensity": modified_params["shock_intensity"]
            })
            
            # Apply price and reward pool shocks with intensity scaling
            if "token_price_multiplier" in modified_params:
                intensity_factor = 1 + shock_effects["shock_intensity"] * 0.2
                token_price *= modified_params["token_price_multiplier"] * intensity_factor
            
            if "reward_pool_modifier" in modified_params:
                intensity_factor = 1 + shock_effects["shock_intensity"] * 0.15
                reward_pool *= modified_params["reward_pool_modifier"] * intensity_factor

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
            # Calculate market saturation
            market_saturation = users / carrying_capacity
            
            # Adjust growth parameters based on saturation
            adjusted_steepness = growth_steepness * (1 - market_saturation)
            adjusted_midpoint = max(midpoint_month, current_month * (1 - market_saturation))
            
            # Calculate logistic growth with adjusted parameters
            logistic_user_count = logistic_growth(
                current_month=month,
                carrying_capacity=carrying_capacity,
                initial_base_users=base_users,
                growth_steepness=adjusted_steepness,
                midpoint=adjusted_midpoint
            )
            new_users = int(max(0, logistic_user_count - users))
        else:
            # Linear growth with saturation effect
            market_saturation = users / total_addressable_market
            saturation_factor = 1 - market_saturation
            effective_growth = effective_growth_rate * saturation_factor
            new_users = int(users * effective_growth)

        # Adjust revenue calculations
        # Scale revenue by market penetration factor
        market_penetration_factor = users / total_addressable_market
        platform_revenue *= market_penetration_factor

        # Implement dynamic churn rates
        base_churn = np.clip(inactivity_rate * (1 + market_saturation), 0.03, 0.12)
        if effective_growth_rate > 0.1:  # If growth rate is high
            base_churn *= 1.5  # Increase churn to balance growth

        # Ensure competitor impacts are realistic
        competitor_impact = 1.0
        for i in range(num_competitors):
            competitor_growth = competitor_growth_rates[i] if isinstance(competitor_growth_rates, list) else competitor_growth_rates
            competitor_attract = competitor_attractiveness[i] if isinstance(competitor_attractiveness, list) else competitor_attractiveness
            competitor_strength = competitor_attract * (1 + competitor_growth)
            competitor_impact *= (1 - competitor_strength)

        # Apply competitor impact to growth
        effective_growth = effective_growth_rate * competitor_impact

        # Enhanced competition and churn effects
        churned_users = int(users * base_churn)

        # Calculate competitor effects with market saturation
        competitor_churn = 0
        for i in range(num_competitors):
            effect = calculate_competitor_impact(
                users,
                token_price,
                initial_token_price,
                competitor_attractiveness[i],
                market_saturation
            )
            competitor_churn += int(users * effect)

        # Apply total churn
        churned_users += competitor_churn

        # Update total users with enhanced bounds checking
        previous_users = users
        users += new_users - churned_users

        # Apply stricter bounds as we approach capacity
        max_users = int(carrying_capacity * 0.95)  # Cap at 95% of TAM
        if users > max_users:
            excess_users = users - max_users
            churn_adjustment = int(excess_users * 0.2)  # Gradually reduce excess
            users = max_users - churn_adjustment

        users = max(base_users * 0.5, users)  # Maintain minimum user base
        
        # Calculate actual growth rate for metrics
        actual_growth_rate = (users - previous_users) / previous_users if previous_users > 0 else 0

        # Update segment counts each month
        segment_counts = {
            segment: int(users * data["proportion"]) for segment, data in user_segments.items()
        }

        # Calculate segment counts with dynamic proportions based on growth
        def calculate_segment_distribution(users, previous_segment_counts, user_segments):
            """Calculate segment counts with growth-based transitions."""
            new_segment_counts = {}
            
            # Calculate base distribution
            base_counts = {
                segment: int(users * data["proportion"])
                for segment, data in user_segments.items()
            }
            
            # Adjust for user growth and segment transitions
            if previous_segment_counts:
                for segment in user_segments:
                    # Calculate segment growth
                    segment_growth = base_counts[segment] - previous_segment_counts[segment]
                    
                    if segment_growth > 0:
                        # For positive growth, apply segment-specific retention
                        retention_rate = user_segments[segment]["churn_resistance"]
                        new_segment_counts[segment] = int(
                            previous_segment_counts[segment] * retention_rate +
                            segment_growth
                        )
                    else:
                        # For negative growth, maintain proportion
                        new_segment_counts[segment] = base_counts[segment]
            else:
                new_segment_counts = base_counts
            
            # Ensure total matches user count
            total_in_segments = sum(new_segment_counts.values())
            if total_in_segments != users:
                # Adjust largest segment to match total
                largest_segment = max(new_segment_counts, key=new_segment_counts.get)
                new_segment_counts[largest_segment] += (users - total_in_segments)
            
            return new_segment_counts

        # Update the segment counts calculation in the simulation loop
        previous_segment_counts = segment_counts.copy() if 'segment_counts' in locals() else None
        segment_counts = calculate_segment_distribution(users, previous_segment_counts, user_segments)

        # Calculate segment-specific metrics
        segment_metrics = {
            segment: {
                'users': segment_counts[segment],
                'contributions': segment_counts[segment] * data["contribution_multiplier"],
                'lookups': (
                    segment_counts[segment] * 
                    customers_per_user * 
                    (lookup_frequency / 12) * 
                    data["lookup_multiplier"]
                ),
                'premium_users': int(
                    segment_counts[segment] * 
                    premium_adoption * 
                    data["premium_adoption_multiplier"]
                )
            }
            for segment, data in user_segments.items()
        }

        # Calculate total contributions with proper scaling
        total_contributions = sum(
            metrics['contributions'] 
            for metrics in segment_metrics.values()
        )

        # Calculate monthly spending with segment-specific rates
        monthly_spending = sum(
            metrics['lookups'] * search_fee
            for metrics in segment_metrics.values()
        )

        # Calculate premium spending with proper segment scaling
        premium_spending = sum(
            metrics['premium_users'] * 10  # Base premium cost
            for metrics in segment_metrics.values()
        )

        # Calculate monthly token spending
        monthly_token_spending = monthly_spending + premium_spending
        
        # Update cumulative token spending
        total_tokens_spent += monthly_token_spending

        # Update monthly results with proper token spending metrics
        monthly_results.append({
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
            "Power User Contribution": segment_metrics["power"]["contributions"],
            "Power User Premium": segment_metrics["power"]["premium_users"],
                "Regular Users": segment_counts["regular"],
            "Regular User Contribution": segment_metrics["regular"]["contributions"],
            "Regular User Premium": segment_metrics["regular"]["premium_users"],
                "Casual Users": segment_counts["casual"],
            "Casual User Contribution": segment_metrics["casual"]["contributions"],
            "Casual User Premium": segment_metrics["casual"]["premium_users"],
            "Total Contributions": total_contributions,
            "Premium Users": sum(metrics['premium_users'] for metrics in segment_metrics.values()),
                "Total Tokens Staked": total_tokens_staked,
                "Reward Pool": reward_pool,
                "Token Sales Revenue": platform_revenue,
                "Actual Growth Rate": actual_growth_rate,
                "Target Growth Rate": effective_growth_rate if not logistic_enabled else None,
                "Distance to Carrying Capacity": carrying_capacity - users if logistic_enabled else None,
                "Shock Event": shock_event_description,
            "Monthly Spending": monthly_spending + premium_spending,  # Add monthly spending
            "Cumulative Spending": total_tokens_spent,  # Add cumulative spending
            "Monthly Token Spending": monthly_token_spending,
            "Premium Spending": premium_spending,
            "Search Spending": monthly_spending,
        })

        # Update reward rate with decay
        reward *= (1 - reward_decay_rate)
        total_tokens_earned += monthly_spending + premium_spending

        # Track total tokens spent
        total_tokens_spent += monthly_spending + premium_spending

        # 6. Transaction Fees and Revenue
        # Calculate base transaction fees in tokens
        base_transaction_fees = (monthly_spending + premium_spending) * transaction_fee_rate
        
        # Convert fees to revenue based on token price
        transaction_revenue = base_transaction_fees * token_price
        fiat_transaction_revenue = transaction_revenue
        
        # Calculate fiat revenue from monthly spending
        monthly_fiat_revenue = monthly_spending * token_price * 0.75
        fiat_premium_revenue = monthly_fiat_revenue
        
        # Update platform revenue
        platform_revenue = fiat_transaction_revenue + fiat_premium_revenue

        # Calculate token revenue streams
        search_spending = monthly_spending  # From search fees
        premium_spending_total = premium_spending  # From premium features
        
        # Update treasury with received tokens
        token_search_revenue = search_spending
        token_premium_revenue = premium_spending_total
        treasury_tokens += (token_search_revenue + token_premium_revenue)

        # Calculate potential fiat value (not actual revenue, just for reference)
        potential_fiat_value = (search_spending + premium_spending_total) * token_price
        
        # Update the monthly results to show all revenue streams:
        monthly_results[-1].update({
            "Potential Fiat Value ($)": potential_fiat_value,
            "Token Search Revenue": token_search_revenue,
            "Token Premium Revenue": token_premium_revenue,
            "Treasury Tokens": treasury_tokens,
        })

        # 7. Reward Pool Management
        # Calculate base rewards per user based on activity
        base_reward_per_user = reward * (1 - reward_decay_rate)
        
        # Scale rewards based on user activity and contributions
        activity_multiplier = (monthly_spending + premium_spending) / (users * search_fee)
        activity_multiplier = np.clip(activity_multiplier, 0.5, 2.0)  # Bound multiplier
        
        # Calculate total monthly rewards with user scaling
        monthly_rewards = min(
            total_contributions * base_reward_per_user * activity_multiplier,  # Activity-based rewards
            reward_pool * (0.05 + (0.05 * (users / base_users)))  # Dynamic pool drain rate
        )
        
        # Calculate revenue-based pool replenishment
        user_scaled_revenue = transaction_revenue * (users / base_users)
        reward_pool_addition = min(
            user_scaled_revenue * reward_pool_share,
            reward_pool_size * 0.2  # Cap monthly addition at 20% of initial size
        )
        
        # Update reward pool with proper scaling
        reward_pool = max(0, reward_pool - monthly_rewards)  # Remove rewards
        reward_pool += reward_pool_addition  # Add scaled revenue share
        
        # Apply dynamic pool cap based on user growth
        user_growth_factor = max(1.0, np.log1p(users / base_users))
        max_pool_size = reward_pool_size * user_growth_factor
        reward_pool = min(reward_pool, max_pool_size)

        # 8. Token Burning
        # Calculate user-scaled burn rate
        effective_burn_rate = burn_rate * (1 + np.log1p(users / base_users) * 0.2)
        
        # Calculate burn from rewards with user scaling
        reward_burn = min(
            monthly_rewards * effective_burn_rate,
            monthly_rewards * 0.05  # Cap at 5%
        )
        
        # Calculate burn from excess rewards relative to user base
        excess_rewards = max(0, monthly_rewards - (transaction_revenue / token_price))
        excess_burn = excess_rewards * min(0.15, 0.1 * (users / base_users))
        
        # Calculate revenue burn with user scaling
        revenue_burn = (transaction_revenue * burn_share) / token_price
        revenue_burn *= (1 + np.log1p(users / base_users) * 0.1)
        
        # Total burn calculation with user-based cap
        max_burn = total_tokens_earned * (0.3 + (0.2 * (users / base_users)))
        total_burn = min(
            reward_burn + revenue_burn + excess_burn,
            max_burn  # Dynamic burn cap
        )

        # Update token totals
        total_tokens_burned += total_burn
        total_tokens_earned = max(0, total_tokens_earned - total_burn)

        # 9. Staking with user correlation
        # Calculate dynamic APR based on user growth and pool health
        pool_ratio = reward_pool / reward_pool_size
        user_ratio = users / base_users
        effective_apr = staking_apr * pool_ratio * min(2.0, np.sqrt(user_ratio))
        monthly_staking_apr = effective_apr / 12
        
        # Calculate staking rewards with user-based cap
        max_staking_rewards = reward_pool * (0.02 * min(2.0, np.sqrt(user_ratio)))
        staking_rewards = min(
            total_tokens_staked * monthly_staking_apr,
            max_staking_rewards
        )

        # Update staking and reward pool
        reward_pool = max(0, reward_pool - staking_rewards)
        total_tokens_earned += staking_rewards

        # Calculate new staking amount
        base_staking_amount = users * (token_price / initial_token_price) * 0.1
        new_tokens_staked = min(
            base_staking_amount * shock_effects["staking_modifier"],
            total_tokens_earned * 0.3  # Cap at 30% of earned tokens
        )

        # Update staking totals
        total_tokens_staked = max(0, total_tokens_staked + new_tokens_staked)
        total_tokens_earned = max(0, total_tokens_earned - new_tokens_staked)

        # 9. Market Sentiment
        # Calculate transaction volume
        current_transaction_volume = monthly_spending + premium_spending
        transaction_volume_change = (
            (current_transaction_volume - previous_transaction_volume) / 
            previous_transaction_volume if previous_transaction_volume > 0 else 0
        )
        
        # Calculate user growth rate
        current_growth_rate = (users - previous_users) / previous_users if previous_users > 0 else 0
        
        # Calculate staked ratio
        staked_ratio = total_tokens_staked / total_tokens_earned if total_tokens_earned > 0 else 0
        
        # Calculate price change
        if len(price_history) > 0:
            current_price_change = (token_price - price_history[-1]) / price_history[-1]
            price_change_history.append(current_price_change)
        else:
            current_price_change = 0
        
        # Update market sentiment with history
        market_sentiment = simulate_market_sentiment(
            market_sentiment,
            market_volatility,
            market_trend,
            current_growth_rate,
            current_price_change,
            transaction_volume_change,
            sentiment_history
        )
        
        # Store sentiment for history
        sentiment_history.append(market_sentiment)
        
        # Trim histories to keep last 12 months
        if len(sentiment_history) > 12:
            sentiment_history = sentiment_history[-12:]
        if len(price_change_history) > 12:
            price_change_history = price_change_history[-12:]
            
        # Calculate smoothed metrics for token price
        smoothed_sentiment = np.mean(sentiment_history[-3:]) if len(sentiment_history) >= 3 else market_sentiment
        smoothed_growth = np.mean([current_growth_rate] + price_change_history[-2:]) if price_change_history else current_growth_rate
        
        # Calculate token price with smoothed inputs
        token_price = calculate_token_price(
            initial_token_price,
            total_tokens_earned,
            total_tokens_spent,
            price_elasticity,
            smoothed_sentiment,  # Use smoothed sentiment
            smoothed_growth,     # Use smoothed growth
            current_transaction_volume,
            staked_ratio,
            current_month,
            price_history
        )
        
        # Store values for next iteration
        previous_transaction_volume = current_transaction_volume
        price_history.append(token_price)

        # 10. Platform Revenue (moved up from previous position 12)
        platform_revenue += monthly_spending * token_price * 0.75

        # Apply persistent shock effects
        if shock_effects["sentiment_persistence"] > 0:
            recovery_factor = (shock_effects["sentiment_persistence"] / 4) ** 1.5
            market_sentiment += shock_effects["cascade_effects"]["sentiment_impact"] * recovery_factor
            shock_effects["sentiment_persistence"] -= 1

        if shock_effects["growth_shock_recovery"] > 0:
            recovery_factor = (shock_effects["growth_shock_recovery"] / 6) ** 2
            effective_growth_rate *= (1 + shock_effects["cascade_effects"]["user_impact"] * recovery_factor)
            shock_effects["growth_shock_recovery"] -= 1

        # Modify competitor effects
        competitor_churn = 0
        for i in range(num_competitors):
            effect = calculate_competitor_impact(
                users,
                token_price,
                initial_token_price,
                competitor_attractiveness[i] * shock_effects["competitor_multiplier"],
                market_saturation
            )
            competitor_churn += int(users * effect)
        
        # Gradually reset shock effects
        shock_effects["competitor_multiplier"] = max(1.0, shock_effects["competitor_multiplier"] * 0.9)
        shock_effects["staking_modifier"] = 1.0 + (shock_effects["staking_modifier"] - 1.0) * 0.8

        # Fix the indentation in the user segments section
        if shock_effects["segment_impact"]:
            # Adjust user segments based on shock effects
            for segment in user_segments:
                # Apply shock modifiers to each segment
                segment_shock = shock_effects.get(f"{segment}_modifier", 1.0)
                user_segments[segment]["proportion"] *= segment_shock
            
            # Renormalize proportions
            total_proportion = sum(segment["proportion"] for segment in user_segments.values())
            if total_proportion > 0:
                for segment in user_segments:
                    user_segments[segment]["proportion"] /= total_proportion

    return pd.DataFrame(monthly_results)

# --- Streamlit UI ---
st.title("PG Tokenomics Simulator")

# Add at the beginning of your sidebar parameters, before other user metrics
st.sidebar.header("🎯 Market Size Parameters")

# Total Addressable Market (TAM) with slider
total_addressable_market = st.sidebar.number_input(
    "Total Addressable Market",
    min_value=1000,
    max_value=10000000,
    value=1000000,
    step=1000,
    help="Total potential users in the market"
)
log_message(f"TAM set to: {total_addressable_market}", debug_only=True)

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

# Calculate and display target percentage
target_percentage = (total_users_target / total_addressable_market) * 100
st.sidebar.markdown(
    f"""
    Target User Growth:
    - Current Target: **{format_number(total_users_target)} users**
    - Represents **{target_percentage:.1f}%** of Total Addressable Market ({format_number(total_addressable_market)} users)
    """
)

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

# User Growth Rate Slider with min/max boxes
growth_rate = create_slider_with_range(
    "User Growth Rate (%/month)",
    default_min=0.0,
    default_max=20.0,
    default_value=6.0,
    step=0.1,
    format="%.1f",
    key_prefix="growth_rate",
    help_text="Monthly user growth rate as a percentage"
)
log_message(f"UI Input - Growth Rate: {growth_rate}%", debug_only=True)

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

reward_pool_share = create_slider_with_range(
    "Reward Pool Share (%)",
    0.0,
    100.0,
    20.0,  # Default 20%
    1.0,
    key_prefix="s2",
    help_text="Percentage of revenue allocated to reward pool (recommended: 15-25%)."
)

burn_share = create_slider_with_range(
    "Burn Share (%)",
    0.0,
    100.0,
    10.0,  # Default 10%
    1.0,
    key_prefix="s2",
    help_text="Percentage of revenue allocated to token burning (recommended: 5-15%)."
)

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

# --- Section 3: Platform Activity Parameters ---

st.sidebar.header("3. Platform Activity Parameters")

contribution_cap = create_slider_with_range(
    "Contribution Cap",
    100,
    5000,
    750,  # Default 750
    50,
    key_prefix="s3",
    help_text="Maximum number of contributions allowed per user (recommended: 500-1000)."
)

transaction_fee_rate = create_slider_with_range(
    "Transaction Fee Rate (%)",
    0.0,
    10.0,
    4.0,  # Default 4%
    0.1,
    key_prefix="s3",
    help_text="Percentage fee charged on token transactions (recommended: 3-5%)."
)

customers_per_user = create_slider_with_range(
    "Customers per User",
    1,
    100,
    50,  # Default 50
    1,
    key_prefix="s3",
    help_text="Average number of customers each user manages."
)

new_customers_per_user = create_slider_with_range(
    "New Customers per User (Monthly)",
    1,
    20,
    7,  # Default 7
    1,
    key_prefix="s3",
    help_text="Average number of new customers each user adds per month."
)

line_items_per_customer = create_slider_with_range(
    "Line Items per Customer",
    1,
    5000,
    75,  # Default 75 items
    1,
    key_prefix="s3",
    help_text="Monthly line items per customer (recommended: 50-100)."
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

# Log the captured initial token price
log_message(f"Captured Initial Token Price: {initial_token_price}", debug_only=True)

price_elasticity = create_slider_with_range(
    "Price Elasticity",
    0.1,
    2.0,
    0.4,  # Default 0.4
    0.1,
    key_prefix="s4",
    help_text="Token price sensitivity to supply/demand (recommended: 0.3-0.5)."
)

# Log the captured price elasticity
log_message(f"Captured Price Elasticity: {price_elasticity}", debug_only=True)

initial_market_sentiment = create_slider_with_range(
    "Initial Market Sentiment",
    0.5,
    1.5,
    1.0,  # Default neutral
    0.1,
    key_prefix="s4",
    help_text="Starting market sentiment (1.0 is neutral)."
)

# Log the captured initial market sentiment
log_message(f"Captured Initial Market Sentiment: {initial_market_sentiment}", debug_only=True)

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

# Log the captured market volatility
log_message(f"Captured Market Volatility: {market_volatility}", debug_only=True)

market_trend = create_slider_with_range(
    "Market Trend",
    -0.1,
    0.1,
    0.0,  # Default neutral
    0.01,
    key_prefix="s4",
    help_text="Long-term market sentiment trend (0.0 is neutral)."
)

# Log the captured market trend
log_message(f"Captured Market Trend: {market_trend}", debug_only=True)

# --- Competition Parameters ---
num_competitors = st.sidebar.number_input("Number of Competitors", 0, 10, 3)
log_message(f"Captured Number of Competitors: {num_competitors}", debug_only=True)

competitor_growth_rates = [0.04] * num_competitors  # Default 4% growth
competitor_attractiveness = [0.02] * num_competitors  # Default 0.02 attractiveness

# Log the captured competitor growth rates and attractiveness
log_message(f"Captured Competitor Growth Rates: {competitor_growth_rates}", debug_only=True)
log_message(f"Captured Competitor Attractiveness: {competitor_attractiveness}", debug_only=True)

# --- Simulation Parameters ---
months = st.sidebar.number_input("Simulation Duration (Months)", 12, 120, 36)
log_message(f"Captured Simulation Duration: {months}", debug_only=True)

# Temporarily disable shock events for testing
shock_events = None

# --- Run Simulation ---
log_message(f"Pre-simulation - Growth Rate (after /100): {growth_rate/100.0}", debug_only=True)
results = simulate_tokenomics(
    initial_reward=initial_reward,
    initial_search_fee=initial_search_fee,
    growth_rate=growth_rate / 100.0,   # Convert 6.0 => 0.06
    line_items_per_customer=line_items_per_customer,
    initial_lookup_frequency=initial_lookup_frequency,
    reward_decay_rate=reward_decay_rate / 100.0,  # 1.5 => 0.015
    contribution_cap=contribution_cap,
    initial_premium_adoption=initial_premium_adoption / 100.0,
    inactivity_rate=inactivity_rate / 100.0,  # 4.0 => 0.04
    months=months,
    base_users=base_users,  # truly from user input
    logistic_enabled=False, # If you want no logistic growth
    customers_per_user=customers_per_user,
    new_customers_per_user=new_customers_per_user,
    initial_token_price=initial_token_price,
    price_elasticity=price_elasticity,
    burn_rate=burn_rate / 100.0,  # 4.0 => 0.04
    initial_market_sentiment=initial_market_sentiment,
    market_volatility=market_volatility,
    market_trend=market_trend,
    staking_apr=staking_apr / 100.0,
    reward_pool_size=reward_pool_size,
    num_competitors=num_competitors,
    competitor_growth_rates=competitor_growth_rates,
    competitor_attractiveness=competitor_attractiveness,
    transaction_fee_rate=transaction_fee_rate / 100.0,
    reward_pool_share=reward_pool_share / 100.0,
    burn_share=burn_share / 100.0,
    token_purchase_threshold=token_purchase_threshold,
    token_purchase_amount=token_purchase_amount,
    token_sale_price=token_sale_price,
    total_users_target=total_users_target,
    total_addressable_market=total_addressable_market,
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

# --- Main Dashboard ---
st.markdown("## Platform Performance Dashboard")

# --- Revenue Section (First and Most Prominent) ---
st.markdown("### Token & Fiat Metrics")

# First row - Token Metrics
token_cols = st.columns(4)

with token_cols[0]:
    treasury_balance = results['Treasury Tokens'].iloc[-1]
    prev_treasury = results['Treasury Tokens'].iloc[-2]
    treasury_growth = ((treasury_balance / prev_treasury) - 1) * 100 if prev_treasury > 0 else 0
    st.metric(
        "Treasury Balance",
        f"{format_number(treasury_balance)} tokens",
        f"{treasury_growth:.1f}% MoM",
        help="Total tokens in platform treasury"
    )

with token_cols[1]:
    potential_value = results['Potential Fiat Value ($)'].iloc[-1]
    prev_value = results['Potential Fiat Value ($)'].iloc[-2]
    value_growth = ((potential_value / prev_value) - 1) * 100 if prev_value > 0 else 0
    st.metric(
        "Potential Treasury Value",
        format_currency(potential_value),
        f"{value_growth:.1f}% MoM",
        help="Current fiat value of treasury tokens (if sold)"
    )

with token_cols[2]:
    token_search = results['Token Search Revenue'].iloc[-1]
    prev_search = results['Token Search Revenue'].iloc[-2]
    search_growth = ((token_search / prev_search) - 1) * 100 if prev_search > 0 else 0
    st.metric(
        "Search Tokens",
        f"{format_number(token_search)} tokens",
        f"{search_growth:.1f}% MoM",
        help="Tokens earned from search fees"
    )

with token_cols[3]:
    token_premium = results['Token Premium Revenue'].iloc[-1]
    prev_premium = results['Token Premium Revenue'].iloc[-2]
    premium_growth = ((token_premium / prev_premium) - 1) * 100 if prev_premium > 0 else 0
    st.metric(
        "Premium Tokens",
        f"{format_number(token_premium)} tokens",
        f"{premium_growth:.1f}% MoM",
        help="Tokens earned from premium features"
    )

# Second row - Fiat Value Metrics
st.markdown("#### Monthly Fiat Revenue")
fiat_cols = st.columns(3)

with fiat_cols[0]:
    # Current month search revenue
    search_revenue = token_search * token_sale_price
    prev_search_revenue = results['Token Search Revenue'].iloc[-2] * token_sale_price
    search_revenue_growth = ((search_revenue / prev_search_revenue) - 1) * 100 if prev_search_revenue > 0 else 0
    
    # Average search revenue
    avg_search_revenue = np.mean(results['Token Search Revenue'] * token_sale_price)
    
    st.metric(
        "Search Revenue",
        format_currency(search_revenue),
        f"{search_revenue_growth:.1f}% MoM",
        help=f"Monthly revenue from users buying search tokens (Avg: {format_currency(avg_search_revenue)})"
    )

with fiat_cols[1]:
    # Current month premium revenue
    premium_revenue = token_premium * token_sale_price
    prev_premium_revenue = results['Token Premium Revenue'].iloc[-2] * token_sale_price
    premium_revenue_growth = ((premium_revenue / prev_premium_revenue) - 1) * 100 if prev_premium_revenue > 0 else 0
    
    # Average premium revenue
    avg_premium_revenue = np.mean(results['Token Premium Revenue'] * token_sale_price)
    
    st.metric(
        "Premium Revenue",
        format_currency(premium_revenue),
        f"{premium_revenue_growth:.1f}% MoM",
        help=f"Monthly revenue from users buying premium tokens (Avg: {format_currency(avg_premium_revenue)})"
    )

with fiat_cols[2]:
    # Current month total revenue
    total_revenue = search_revenue + premium_revenue
    prev_total_revenue = prev_search_revenue + prev_premium_revenue
    total_revenue_growth = ((total_revenue / prev_total_revenue) - 1) * 100 if prev_total_revenue > 0 else 0
    
    # Average total revenue
    avg_total_revenue = avg_search_revenue + avg_premium_revenue
    
    st.metric(
        "Total Revenue",
        format_currency(total_revenue),
        f"{total_revenue_growth:.1f}% MoM",
        help=f"Total monthly revenue from token sales (Avg: {format_currency(avg_total_revenue)})"
    )

# Revenue Metrics Section
st.markdown("### Platform Revenue")
revenue_cols = st.columns(4)

with revenue_cols[0]:
    # Monthly Recurring Revenue (MRR)
    current_mrr = search_revenue + premium_revenue
    prev_mrr = prev_search_revenue + prev_premium_revenue
    mrr_growth = ((current_mrr / prev_mrr) - 1) * 100 if prev_mrr > 0 else 0
    
    st.metric(
        "Monthly Recurring Revenue",
        format_currency(current_mrr),
        f"{mrr_growth:.1f}% MoM",
        help="Total monthly recurring revenue from all sources"
    )

with revenue_cols[1]:
    # Annual Recurring Revenue (ARR)
    arr = current_mrr * 12
    prev_arr = prev_mrr * 12
    arr_growth = ((arr / prev_arr) - 1) * 100 if prev_arr > 0 else 0
    
    st.metric(
        "Annual Run Rate",
        format_currency(arr),
        f"{arr_growth:.1f}% YoY",
        help="Annual revenue run rate based on current MRR"
    )

with revenue_cols[2]:
    # Revenue Growth Rate (3-month average)
    last_3_months = results.tail(3)
    avg_growth_rate = np.mean([
        ((row['Token Search Revenue'] + row['Token Premium Revenue']) /
         (results.iloc[i-1]['Token Search Revenue'] + results.iloc[i-1]['Token Premium Revenue']) - 1) * 100
        for i, row in last_3_months.iterrows()
        if i > 0
    ])
    
    st.metric(
        "Growth Rate (3mo avg)",
        f"{avg_growth_rate:.1f}%",
        help="Average monthly revenue growth rate over the last 3 months"
    )

with revenue_cols[3]:
    # Revenue per User
    total_users = results['Users'].iloc[-1]
    revenue_per_user = current_mrr / total_users if total_users > 0 else 0
    prev_revenue_per_user = prev_mrr / results['Users'].iloc[-2] if results['Users'].iloc[-2] > 0 else 0
    rpu_growth = ((revenue_per_user / prev_revenue_per_user) - 1) * 100 if prev_revenue_per_user > 0 else 0
    
    st.metric(
        "Revenue per User",
        format_currency(revenue_per_user),
        f"{rpu_growth:.1f}% MoM",
        help="Average monthly revenue generated per active user"
    )

# Growth & Efficiency Metrics
st.markdown("### Growth & Efficiency Metrics")
growth_cols = st.columns(4)

with growth_cols[0]:
    # Customer Acquisition Cost (CAC)
    # Calculate new users (3-month rolling average)
    new_users_series = results['Users'].diff().tail(3)
    avg_new_users = new_users_series[new_users_series > 0].mean()
    
    # Calculate marketing spend based on token purchase amount
    token_purchase_amount = 15.0  # From parameters
    marketing_percent = 0.20  # Assume 20% of revenue goes to marketing
    
    # Calculate average revenue per new user and marketing spend
    avg_revenue_per_new_user = token_purchase_amount * token_sale_price
    marketing_spend = avg_revenue_per_new_user * marketing_percent
    
    # Calculate CAC (marketing spend per new user)
    if not pd.isna(avg_new_users) and avg_new_users > 0:
        cac = marketing_spend
    else:
        cac = marketing_spend  # Fallback when no new users
        
    # Calculate previous period values
    prev_new_users = results['Users'].diff().iloc[-2] if len(results) > 1 else avg_new_users
    prev_cac = marketing_spend  # CAC should be constant if based on fixed token purchase amount
    cac_change = 0  # No change since CAC is fixed
    
    st.metric(
        "Customer Acquisition Cost",
        format_currency(cac),
        f"{cac_change:.1f}% MoM",
        help="Cost to acquire a new user (Based on 20% of initial token purchase amount)"
    )

with growth_cols[1]:
    # Lifetime Value (LTV)
    avg_lifetime = 12  # Assumed average user lifetime in months
    ltv = revenue_per_user * avg_lifetime
    prev_ltv = prev_revenue_per_user * avg_lifetime
    ltv_change = ((ltv / prev_ltv) - 1) * 100 if prev_ltv > 0 else 0
    
    st.metric(
        "Lifetime Value",
        format_currency(ltv),
        f"{ltv_change:.1f}% MoM",
        help="Projected total revenue from an average user (Assumption: Average user lifetime is 12 months)"
    )

with growth_cols[2]:
    # LTV/CAC Ratio
    if cac and cac > 0:
        ltv_cac_ratio = ltv / cac
        if prev_cac and prev_cac > 0:
            prev_ltv_cac = prev_ltv / prev_cac
            ratio_change = ((ltv_cac_ratio / prev_ltv_cac) - 1) * 100 if prev_ltv_cac > 0 else 0
        else:
            ratio_change = 0
    else:
        ltv_cac_ratio = 0
        ratio_change = 0
    
    st.metric(
        "LTV/CAC Ratio",
        f"{ltv_cac_ratio:.2f}x" if ltv_cac_ratio > 0 else "N/A",
        f"{ratio_change:.1f}% MoM" if ratio_change != 0 else None,
        help="Ratio of lifetime value to acquisition cost (Based on estimated CAC and LTV. Target: >3x for healthy growth)"
    )

with growth_cols[3]:
    current_users = results['Users'].iloc[-1]
    prev_users = results['Users'].iloc[-2] if len(results) > 1 else current_users

    market_penetration = (current_users / total_addressable_market) * 100
    target_penetration = (total_users_target / total_addressable_market) * 100
    prev_penetration = (prev_users / total_addressable_market) * 100
    penetration_change = market_penetration - prev_penetration

    st.metric(
        "Market Penetration",
        f"{market_penetration:.1f}%",
        f"{penetration_change:+.1f}pp",
        help=f"""
        Current Market Share: {market_penetration:.1f}% of TAM ({format_number(total_addressable_market)} users)
        Target: {target_penetration:.1f}% ({format_number(total_users_target)} users)
        Monthly Growth Rate: {growth_rate:.1f}%
        """
    )

    # Add debug logs
    log_message(f"Current Users: {current_users}", debug_only=True)
    log_message(f"Previous Users: {prev_users}", debug_only=True)
    log_message(f"Target Users: {total_users_target}", debug_only=True)
    log_message(f"TAM: {total_addressable_market}", debug_only=True)
    log_message(f"Market Penetration: {market_penetration:.2f}%", debug_only=True)
    log_message(f"Target Penetration: {target_penetration:.2f}%", debug_only=True)
    log_message(f"Penetration Change: {penetration_change:.2f}pp", debug_only=True)

# Token Economics
st.markdown("### Token Economics")
token_cols = st.columns(4)

with token_cols[0]:
    # Token Velocity
    monthly_token_volume = results['Monthly Token Spending'].iloc[-1]
    token_supply = treasury_balance
    token_velocity = monthly_token_volume / token_supply if token_supply > 0 else 0
    prev_velocity = (results['Monthly Token Spending'].iloc[-2] / 
                    results['Treasury Tokens'].iloc[-2]) if results['Treasury Tokens'].iloc[-2] > 0 else 0
    velocity_change = ((token_velocity / prev_velocity) - 1) * 100 if prev_velocity > 0 else 0
    
    st.metric(
        "Token Velocity",
        f"{token_velocity:.2f}x",
        f"{velocity_change:.1f}% MoM",
        help="Rate at which tokens change hands (Calculated from actual monthly volume / treasury supply)"
    )

with token_cols[1]:
    # Staking Ratio
    staked_tokens = results['Total Tokens Staked'].iloc[-1]
    staking_ratio = (staked_tokens / token_supply) * 100 if token_supply > 0 else 0
    prev_staking = (results['Total Tokens Staked'].iloc[-2] / 
                   results['Treasury Tokens'].iloc[-2]) * 100 if results['Treasury Tokens'].iloc[-2] > 0 else 0
    staking_change = staking_ratio - prev_staking
    
    st.metric(
        "Staking Ratio",
        f"{staking_ratio:.1f}%",
        f"{staking_change:+.1f}pp",
        help=f"Percentage of treasury tokens currently staked (Target: {staking_apr:.1f}% APR)"
    )

with token_cols[2]:
    # Token Holder Growth
    token_holders = int(total_users * 0.8)  # Assumption: 80% of users hold tokens
    prev_holders = int(results['Users'].iloc[-2] * 0.8)
    holder_growth = ((token_holders / prev_holders) - 1) * 100 if prev_holders > 0 else 0
    
    st.metric(
        "Token Holders",
        format_number(token_holders),
        f"{holder_growth:.1f}% MoM",
        help="Estimated number of unique token holders (Assumption: 80% of active users hold tokens)"
    )

with token_cols[3]:
    # Token Retention
    token_retention = ((staked_tokens + monthly_token_volume) / token_supply) * 100 if token_supply > 0 else 0
    prev_retention = ((results['Total Tokens Staked'].iloc[-2] + results['Monthly Token Spending'].iloc[-2]) / 
                     results['Treasury Tokens'].iloc[-2]) * 100 if results['Treasury Tokens'].iloc[-2] > 0 else 0
    retention_change = token_retention - prev_retention
    
    st.metric(
        "Token Retention",
        f"{token_retention:.1f}%",
        f"{retention_change:+.1f}pp",
        help="Percentage of tokens actively used or staked (Calculated from actual staking and spending data)"
    )

# Platform Treasury & Revenue Chart
st.markdown("### Platform Treasury & Revenue")
token_fig = create_dual_axis_chart(results)
st.plotly_chart(token_fig, use_container_width=True)

st.markdown("---")

# Continue with rest of the dashboard (Token Performance, etc.)

# --- Create a container for the floating charts ---
chart_container = st.container()

with chart_container:
    # Header section with key metrics
    st.markdown("### Token Performance Dashboard")
    
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
            f"{(user_metrics['current_users'] / total_addressable_market * 100):.1f}%",
            help="Percentage of the total addressable market currently using the platform"
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
        months,
        results
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
        if burn_metrics['monthly_burn'] > 0:
            monthly_burn_formatted = format_number(burn_metrics['monthly_burn'])
            burn_status = "🟢" if burn_metrics['is_sustainable'] else "🔴"
            burn_text = f"{monthly_burn_formatted} {burn_status}"
            burn_delta = f"-{burn_metrics['burn_rate_pct']:.1f}% of initial"
        else:
            monthly_growth_formatted = format_number(burn_metrics['monthly_growth'])
            burn_text = f"+{monthly_growth_formatted} 🟢"
            burn_delta = f"+{burn_metrics['growth_rate_pct']:.1f}% of initial"
        
        st.metric(
            "Monthly Reserve Change",
            burn_text,
            burn_delta,
            help=f"""
            Average monthly change in reserve balance
            Revenue/Burn Ratio: {burn_metrics['revenue_burn_ratio']:.2f}
            Burn Sustainability: {burn_metrics['burn_sustainability']:.1f}%
            """
        )
    
    # Update burn sustainability metrics
    if burn_metrics['monthly_burn'] > 0 and not burn_metrics['is_sustainable']:
        st.warning(f"""
            🔥 **Current burn rate exceeds sustainable levels**
            - Sustainable burn: {format_number(burn_metrics['sustainable_burn'])} tokens/month
            - Current burn: {format_number(burn_metrics['monthly_burn'])} tokens/month
            - Sustainability gap: {abs(burn_metrics['burn_sustainability']):.1f}%
        """)
    elif burn_metrics['monthly_growth'] > 0:
        st.success(f"""
            📈 **Reserve is growing**
            - Monthly growth: {format_number(burn_metrics['monthly_growth'])} tokens/month
            - Growth rate: {burn_metrics['growth_rate_pct']:.1f}% of initial reserve
        """)
    
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
    st.markdown("### Detailed Analysis")
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

    # --- Enhanced Reserve Health Metrics section ---

    # Calculate comprehensive reserve metrics
    reserve_metrics = calculate_reserve_metrics(results, reward_pool_size)

    st.markdown("---")
    st.markdown("### Reserve Health Metrics")

    # Display key reserve metrics with modern styling
    reserve_cols = st.columns(4)
    with reserve_cols[0]:
        st.metric(
            label="Current Reserve",
            value=format_currency(results['Reward Pool'].iloc[-1]),
            delta=f"{((results['Reward Pool'].iloc[-1] - reward_pool_size)/reward_pool_size*100):.1f}%",
            help="Current balance in the reward pool"
        )

    with reserve_cols[1]:
        st.metric(
            label="Initial Reserve",
            value=format_currency(reward_pool_size),
            help="Initial reward pool size"
        )

    with reserve_cols[2]:
        st.metric(
            label="Minimum Reserve",
            value=format_currency(results['Reward Pool'].min()),
            delta=f"{(results['Reward Pool'].min()/reward_pool_size*100):.1f}% of initial",
            help="Lowest reserve balance over time"
        )

    with reserve_cols[3]:
        # Reserve Health Status
        health_status, ratio = calculate_reserve_health(
            results['Reward Pool'].iloc[-1],
            reward_pool_size
        )
        st.metric(
            label="Reserve Health",
            value=health_status,
            delta=f"{(ratio*100):.1f}% of initial",
            help="Current reserve health status"
        )

    # Create an enhanced reserve trend chart with shading for different reserve states

    # Prepare data for chart
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

    # Create reserve trend chart with colored bands for different states
    fig = go.Figure()

    # Add Reward Pool line
    fig.add_trace(go.Scatter(
        x=timeline_data['Month'],
        y=timeline_data['Reward Pool'],
        mode='lines',
        name='Reward Pool',
        line=dict(color='#636EFA', width=3)
    ))

    # Add fill between lines to indicate states (using shapes for background colors)
    # Create shapes for different reserve states
    shapes = []

    # Healthy
    shapes.append(dict(
        type='rect',
        xref='x',
        yref='y',
        x0=timeline_data['Month'].min(),
        y0=reserve_metrics['warning_threshold'],
        x1=timeline_data['Month'].max(),
        y1=timeline_data['Reward Pool'].max(),
        fillcolor='rgba(0, 255, 0, 0.1)',  # Light green
        layer='below',
        line_width=0
    ))

    # Warning
    shapes.append(dict(
        type='rect',
        xref='x',
        yref='y',
        x0=timeline_data['Month'].min(),
        y0=reserve_metrics['critical_threshold'],
        x1=timeline_data['Month'].max(),
        y1=reserve_metrics['warning_threshold'],
        fillcolor='rgba(255, 255, 0, 0.1)',  # Light yellow
        layer='below',
        line_width=0
    ))

    # Critical
    shapes.append(dict(
        type='rect',
        xref='x',
        yref='y',
        x0=timeline_data['Month'].min(),
        y0=0,
        x1=timeline_data['Month'].max(),
        y1=reserve_metrics['critical_threshold'],
        fillcolor='rgba(255, 0, 0, 0.1)',  # Light red
        layer='below',
        line_width=0
    ))

    # Add shapes to layout
    fig.update_layout(shapes=shapes)

    # Update layout with modern styling
    fig.update_layout(
        title='Reward Pool Over Time with Reserve Health States',
        xaxis_title='Month',
        yaxis_title='Reward Pool',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=30, t=50, b=60),
        font=dict(size=14),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.1)'
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.1)'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add a summary of time spent in each state with modern UI
    state_percentages = timeline_data['State'].value_counts(normalize=True)*100

    st.markdown("#### Time Spent in Each Reserve State")
    state_cols = st.columns(4)
    with state_cols[0]:
        st.metric(
            label='Healthy',
            value=f"{state_percentages.get('Healthy', 0):.1f}%",
            help='Percentage of time reserve was in Healthy state'
        )
    with state_cols[1]:
        st.metric(
            label='Warning',
            value=f"{state_percentages.get('Warning', 0):.1f}%",
            help='Percentage of time reserve was in Warning state'
        )
    with state_cols[2]:
        st.metric(
            label='Critical',
            value=f"{state_percentages.get('Critical', 0):.1f}%",
            help='Percentage of time reserve was in Critical state'
        )
    with state_cols[3]:
        st.metric(
            label='Deficit',
            value=f"{state_percentages.get('Deficit', 0):.1f}%",
            help='Percentage of time reserve was in Deficit state'
        )

    # Add interactive elements for deeper insights
    st.markdown("#### Interactive Insights")
    st.markdown("Use the chart above to explore how the reserve levels have changed over time. Hover over the lines to see detailed data points.")

    # Update the display section to show recovery metrics
    def display_reserve_metrics(metrics, initial_reserve):
        """Display enhanced reserve metrics including recovery events."""
        st.markdown("### Reserve Health Metrics")
        
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                "Recovery Events",
                f"{metrics['recovery_count']}",
                f"{metrics['recovery_strength']:.1f}% avg strength",
                help="Number of times reserve recovered from warning/critical levels"
            )
        
        with cols[1]:
            time_in_warning = f"{metrics['warning_percentage']:.1f}%"
            st.metric(
                "Time in Warning",
                time_in_warning,
                f"{metrics['months_in_warning']} months",
                help="Percentage of time reserve was below warning threshold"
            )
        
        with cols[2]:
            time_in_critical = f"{metrics['critical_percentage']:.1f}%"
            st.metric(
                "Time in Critical",
                time_in_critical,
                f"{metrics['months_in_critical']} months",
                help="Percentage of time reserve was below critical threshold"
            )
        
        with cols[3]:
            # Calculate stability score based on state percentages
            stability_score = 100 * (1 - (
                metrics['warning_percentage'] + 
                metrics['critical_percentage'] * 2 + 
                metrics['deficit_percentage'] * 3
            ) / 300)  # Weighted impact of different states
            
            # Determine trend based on recent state changes
            if metrics['current_state'] == 'healthy' and metrics['consecutive_months_below'] == 0:
                trend = "Stable"
            elif metrics['consecutive_months_below'] > 0:
                trend = "Declining"
            else:
                trend = "Recovering"
            
            # Set color based on stability score
            stability_color = "🟢" if stability_score > 80 else "🟡" if stability_score > 60 else "🔴"
            
            st.metric(
                "Reserve Stability",
                f"{stability_score:.0f}/100 {stability_color}",
                trend,
                help=f"""
                Reserve Stability Score (higher is better)
                - Warning Time: {metrics['warning_percentage']:.1f}%
                - Critical Time: {metrics['critical_percentage']:.1f}%
                - Current State: {metrics['current_state']}
                - Consecutive Months Below: {metrics['consecutive_months_below']}
                """
            )

        # Add recovery event details if any exist
        if metrics['recovery_events']:
            st.markdown("#### Recovery Events")
            for event in metrics['recovery_events']:
                st.markdown(f"""
                    - Month {event['month']}: Recovered from **{event['from_state']}** state
                    - Duration: {event['duration']} months
                    - Recovery Magnitude: {(event['recovery_magnitude'] * 100):.1f}%
                """)

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
    
    return fig  # Ensuring Git detects changes

# Create an enhanced token activity timeline chart
def create_token_activity_chart(timeline_data, reserve_metrics):
    """Create an enhanced token activity visualization with key metrics and thresholds."""
    
    # Create the main chart with multiple traces
    fig = go.Figure()
    
    # Add reward pool with colored zones for different states
    fig.add_trace(go.Scatter(
        x=timeline_data['Month'],
        y=timeline_data['Reward Pool'],
        name='Reward Pool',
        line=dict(color='#00CC96', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 204, 150, 0.1)'
    ))
    
    # Add token spending trend
    fig.add_trace(go.Scatter(
        x=timeline_data['Month'],
        y=timeline_data['Tokens Spent'],
        name='Tokens Spent',
        line=dict(color='#EF553B', width=2)
    ))
    
    # Add token earning trend
    fig.add_trace(go.Scatter(
        x=timeline_data['Month'],
        y=timeline_data['Tokens Earned'],
        name='Tokens Earned',
        line=dict(color='#636EFA', width=2)
    ))
    
    # Add dynamic threshold lines
    fig.add_trace(go.Scatter(
        x=timeline_data['Month'],
        y=reserve_metrics['dynamic_thresholds']['warning_threshold'],
        name='Warning Threshold',
        line=dict(color='#FFA15A', dash='dash', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=timeline_data['Month'],
        y=reserve_metrics['dynamic_thresholds']['critical_threshold'],
        name='Critical Threshold',
        line=dict(color='#EF553B', dash='dash', width=1)
    ))
    
    # Add recovery events as markers
    if reserve_metrics['recovery_events']:
        recovery_months = [event['month'] for event in reserve_metrics['recovery_events']]
        recovery_values = [timeline_data['Reward Pool'].iloc[month] for month in recovery_months]
        
        fig.add_trace(go.Scatter(
            x=recovery_months,
            y=recovery_values,
            mode='markers',
            name='Recovery Events',
            marker=dict(
                symbol='star',
                size=12,
                color='#00CC96',
                line=dict(width=2, color='white')
            )
        ))
    
    # Update layout with enhanced features
    fig.update_layout(
        title='Token Activity and Reserve Health',
        yaxis_title='Tokens',
        xaxis_title='Month',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.1)'
        ),
        # Add annotations for key metrics
        annotations=[
            dict(
                x=0.02,
                y=1.05,
                xref='paper',
                yref='paper',
                text=f'Stability Score: {reserve_metrics["stability_score"]:.0f}/100',
                showarrow=False
            ),
            dict(
                x=0.02,
                y=1.02,
                xref='paper',
                yref='paper',
                text=f'Recovery Events: {reserve_metrics["recovery_count"]}',
                showarrow=False
            )
        ]
    )
    
    # Add range selector for time window
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=12, label="1y", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig

# Add metrics summary below chart
with st.expander("📊 Token Activity Analysis"):
    metrics_cols = st.columns(3)
    
    with metrics_cols[0]:
        st.metric(
            "Token Velocity",
            f"{(timeline_data['Tokens Spent'].sum() / timeline_data['Tokens Earned'].sum()):.2f}x",
            help="Rate at which tokens are being spent relative to earnings"
        )
    
    with metrics_cols[1]:
        st.metric(
            "Net Token Balance",
            format_number(timeline_data['Tokens Earned'].iloc[-1] - timeline_data['Tokens Spent'].iloc[-1]),
            help="Current difference between total tokens earned and spent"
        )
    
    with metrics_cols[2]:
        burn_rate = (timeline_data['Tokens Burned'].diff() / timeline_data['Tokens Earned'].diff()).mean()
        st.metric(
            "Average Burn Rate",
            f"{burn_rate:.1%}",
            help="Average rate of token burning relative to earnings"
        )

# --- Debugging Section ---
with st.expander("🔍 Debugging & Logging Information"):
    st.write("### Input Parameters")
    st.write(f"User Growth Rate: {growth_rate}")
    st.write(f"Inactivity Rate: {inactivity_rate}")
    st.write(f"Initial User Base: {base_users}")
    st.write(f"Initial Reward: {initial_reward}")
    st.write(f"Reward Decay Rate: {reward_decay_rate}")
    st.write(f"Burn Rate: {burn_rate}")
    st.write(f"Staking APR: {staking_apr}")
    st.write(f"Initial Reward Pool Size: {reward_pool_size}")
    st.write(f"Line Items per Customer: {line_items_per_customer}")
    st.write(f"Contribution Cap: {contribution_cap}")
    st.write(f"Initial Lookups per Customer per Year: {initial_lookup_frequency}")
    st.write(f"Initial Premium Adoption Rate: {initial_premium_adoption}")
    st.write(f"Customers per User: {customers_per_user}")
    st.write(f"New Customers per User: {new_customers_per_user}")
    st.write(f"Initial Search Fee: {initial_search_fee}")
    st.write(f"Transaction Fee Rate: {transaction_fee_rate}")
    st.write(f"Token Purchase Threshold: {token_purchase_threshold}")
    st.write(f"Token Purchase Amount: {token_purchase_amount}")
    st.write(f"Token Sale Price: {token_sale_price}")
    st.write(f"Initial Token Price: {initial_token_price}")
    st.write(f"Price Elasticity: {price_elasticity}")
    st.write(f"Initial Market Sentiment: {initial_market_sentiment}")
    st.write(f"Market Volatility: {market_volatility}")
    st.write(f"Market Trend: {market_trend}")
    st.write(f"Number of Competitors: {num_competitors}")
    st.write(f"Competitor Growth Rates: {competitor_growth_rates}")
    st.write(f"Competitor Attractiveness: {competitor_attractiveness}")
    st.write(f"Simulation Duration: {months}")
    
    st.write("### Simulation Results")
    st.write(results.head())
    
    st.write("### Metrics")
    st.write(f"Reserve Health Status: {status_text}, Ratio: {ratio}")
    st.write(f"Burn Metrics: {burn_metrics}")

    st.write("### Debug Logs")
    for log in st.session_state['debug_logs']:
        st.write(log)

# --- Optimization Section ---
with st.expander("🎯 Optimization Panel", expanded=False):
    # Set Optimization Objectives
    st.write("#### Set Optimization Objectives")
    col1, col2 = st.columns(2)
    
    with col1:
        optimize_objectives = {
            'token_price': st.checkbox('Maximize Token Price', value=True),
            'user_growth': st.checkbox('Maximize User Growth', value=True),
            'reserve_health': st.checkbox('Maximize Reserve Health', value=True),
            'token_velocity': st.checkbox('Optimize Token Velocity', value=False)
        }
    
    with col2:
        objective_weights = {
            'token_price': st.slider('Token Price Weight', 0.0, 1.0, 0.3),
            'user_growth': st.slider('User Growth Weight', 0.0, 1.0, 0.3),
            'reserve_health': st.slider('Reserve Health Weight', 0.0, 1.0, 0.2),
            'token_velocity': st.slider('Token Velocity Weight', 0.0, 1.0, 0.2)
        }

    # Optimization Parameters
    st.write("#### Set Optimization Parameters")
    col3, col4 = st.columns(2)
    
    with col3:
        num_iterations = st.number_input('Number of Iterations', 10, 1000, 100)
        population_size = st.number_input('Population Size', 10, 200, 50)
    
    with col4:
        mutation_rate = st.slider('Mutation Rate', 0.01, 0.5, 0.1)
        convergence_threshold = st.slider('Convergence Threshold', 0.001, 0.1, 0.01)
    
    # Run Optimization
    if st.button('Run Optimization'):
        with st.spinner('Running optimization...'):
            try:
                optimization_results = run_tokenomics_optimization(
                    current_params={
                        'token_price': initial_token_price,
                        'burn_rate': burn_rate,
                        'reward_decay_rate': reward_decay_rate,
                        'staking_apr': staking_apr,
                        'transaction_fee_rate': transaction_fee_rate
                    },
                    objectives=optimize_objectives,
                    weights=objective_weights,
                    num_iterations=num_iterations,
                    population_size=population_size,
                    mutation_rate=mutation_rate,
                    convergence_threshold=convergence_threshold,
                    simulation_params={
                        'initial_reward': initial_reward,
                        'initial_search_fee': initial_search_fee,
                        'growth_rate': growth_rate,
                        'line_items_per_customer': line_items_per_customer,
                        'initial_lookup_frequency': initial_lookup_frequency,
                        'contribution_cap': contribution_cap,
                        'initial_premium_adoption': initial_premium_adoption,
                        'inactivity_rate': inactivity_rate,
                        'months': months,
                        'base_users': base_users,
                        'customers_per_user': customers_per_user,
                        'new_customers_per_user': new_customers_per_user,
                        'price_elasticity': price_elasticity,
                        'initial_market_sentiment': initial_market_sentiment,
                        'market_volatility': market_volatility,
                        'market_trend': market_trend,
                        'reward_pool_size': reward_pool_size,
                        'num_competitors': num_competitors,
                        'competitor_growth_rates': competitor_growth_rates,
                        'competitor_attractiveness': competitor_attractiveness,
                        'token_purchase_threshold': token_purchase_threshold,
                        'token_purchase_amount': token_purchase_amount,
                        'token_sale_price': token_sale_price,
                        'total_users_target': total_users_target,
                        'total_addressable_market': total_addressable_market,
                        'logistic_enabled': True,
                        'carrying_capacity': total_addressable_market,
                        'growth_steepness': 0.25,
                        'midpoint_month': 12,
                        'total_vested_tokens': 100_000,
                        'vest_duration': 12
                    }
                )
                
                st.success('Optimization complete!')
                
                # Display optimization results
                st.write("### Optimization Results")
                st.write("#### Recommended Parameters:")
                for param, value in optimization_results['recommended_params'].items():
                    st.metric(
                        label=param.replace('_', ' ').title(),
                        value=f"{value:.4f}"
                    )
                
                st.write("#### Performance Metrics:")
                st.line_chart(optimization_results['convergence_history'])
                
                st.write("### Detailed Analysis")
                tab1, tab2 = st.tabs(["Parameter Sensitivity", "Trade-offs"])
                
                with tab1:
                    st.write("#### Parameter Sensitivity Analysis")
                    st.write(optimization_results['sensitivity_analysis'])
                
                with tab2:
                    st.write("#### Trade-off Analysis")
                    st.write(optimization_results['trade_offs'])
            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")
