import numpy as np
import pandas as pd

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
    """Simulate tokenomics with given parameters."""
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
            "contribution_multiplier": 3.0,
            "lookup_multiplier": 1.5,
            "premium_adoption_multiplier": 1.5,
            "churn_resistance": 0.8
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
            "contribution_multiplier": 0.3,
            "lookup_multiplier": 0.5,
            "premium_adoption_multiplier": 0.5,
            "churn_resistance": 1.2
        }
    }

    monthly_results = []
    segment_counts = {
        segment: int(users * data["proportion"]) 
        for segment, data in user_segments.items()
    }

    # Initialize simulation parameters
    current_params = {
        "growth_rate": effective_growth_rate,
        "market_sentiment": initial_market_sentiment,
        "inactivity_rate": inactivity_rate,
        "token_price_multiplier": 1.0,
        "reward_pool_modifier": 1.0
    }

    # Initialize tracking variables
    previous_price = initial_token_price
    previous_transaction_volume = 0
    price_history = []
    sentiment_history = []
    price_change_history = []

    # --- Simulation Loop ---
    for month in range(months):
        current_month = month + 1
        
        # Calculate market metrics
        if logistic_enabled:
            market_saturation = users / carrying_capacity
            adjusted_steepness = growth_steepness * (1 - market_saturation)
            adjusted_midpoint = max(midpoint_month, current_month * (1 - market_saturation))
            new_users = int(max(0, (carrying_capacity / (1 + np.exp(-adjusted_steepness * (current_month - adjusted_midpoint)))) - users))
        else:
            market_saturation = users / total_addressable_market
            saturation_factor = 1 - market_saturation
            effective_growth = effective_growth_rate * saturation_factor
            new_users = int(users * effective_growth)

        # Calculate churn
        base_churn = np.clip(inactivity_rate * (1 + market_saturation), 0.03, 0.12)
        churned_users = int(users * base_churn)

        # Update user count
        users += new_users - churned_users
        users = min(users, int(carrying_capacity * 0.95))
        users = max(users, base_users * 0.5)

        # Calculate segment metrics
        segment_counts = {
            segment: int(users * data["proportion"])
            for segment, data in user_segments.items()
        }

        segment_metrics = {
            segment: {
                'users': count,
                'contributions': count * user_segments[segment]["contribution_multiplier"],
                'lookups': (
                    count * 
                    customers_per_user * 
                    (lookup_frequency / 12) * 
                    user_segments[segment]["lookup_multiplier"]
                ),
                'premium_users': int(
                    count * 
                    premium_adoption * 
                    user_segments[segment]["premium_adoption_multiplier"]
                )
            }
            for segment, count in segment_counts.items()
        }

        # Calculate financial metrics
        total_contributions = sum(
            metrics['contributions'] 
            for metrics in segment_metrics.values()
        )

        monthly_spending = sum(
            metrics['lookups'] * search_fee
            for metrics in segment_metrics.values()
        )

        premium_spending = sum(
            metrics['premium_users'] * 10
            for metrics in segment_metrics.values()
        )

        monthly_token_spending = monthly_spending + premium_spending
        total_tokens_spent += monthly_token_spending

        # Update token price
        current_transaction_volume = monthly_token_spending
        transaction_volume_change = (
            (current_transaction_volume - previous_transaction_volume) / 
            previous_transaction_volume if previous_transaction_volume > 0 else 0
        )

        if len(price_history) > 0:
            current_price_change = (token_price - previous_price) / previous_price
            price_change_history.append(current_price_change)
        else:
            current_price_change = 0

        # Calculate market sentiment
        sentiment_factor = 1 + (market_sentiment - 1) * 0.7
        price_factor = 1 + current_price_change * 0.5
        volume_factor = 1 + transaction_volume_change * 0.3

        market_sentiment = market_sentiment * (
            sentiment_factor * price_factor * volume_factor
        )
        market_sentiment = np.clip(market_sentiment, 0.5, 2.0)

        # Update token price
        supply_demand_ratio = total_tokens_spent / total_tokens_earned if total_tokens_earned > 0 else 1
        token_price = initial_token_price * (
            supply_demand_ratio ** price_elasticity * 
            market_sentiment
        )
        token_price = np.clip(token_price, initial_token_price * 0.1, initial_token_price * 10)

        # Store results
        monthly_results.append({
            "Month": current_month,
            "Platform Revenue ($)": platform_revenue,
            "Users": users,
            "Token Price": token_price,
            "Market Sentiment": market_sentiment,
            "Tokens Earned": total_tokens_earned,
            "Tokens Spent": total_tokens_spent,
            "Tokens Burned": total_tokens_burned,
            "Net Token Balance": total_tokens_earned - total_tokens_spent,
            "Search Fee": search_fee,
            "Reward": reward,
            "Power Users": segment_counts["power"],
            "Regular Users": segment_counts["regular"],
            "Casual Users": segment_counts["casual"],
            "Premium Users": sum(metrics['premium_users'] for metrics in segment_metrics.values()),
            "Total Tokens Staked": total_tokens_staked,
            "Reward Pool": reward_pool,
            "Monthly Token Spending": monthly_token_spending,
            "Monthly Spending": monthly_spending + premium_spending,
            "Premium Spending": premium_spending,
            "Search Spending": monthly_spending,
        })

        # Update tracking variables
        previous_price = token_price
        previous_transaction_volume = current_transaction_volume
        price_history.append(token_price)
        sentiment_history.append(market_sentiment)

    return pd.DataFrame(monthly_results)
