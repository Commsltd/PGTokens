import numpy as np
import pandas as pd

def get_token_metrics(results):
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
    ratio = current_reserve / initial_reserve
    
    if ratio >= 0.50:
        return "Healthy 🟢", ratio
    elif ratio >= 0.25:
        return "Warning 🟡", ratio
    else:
        return "Critical 🔴", ratio

def calculate_monthly_burn_metrics(current_reserve, initial_reserve, months, results_df):
    recent_months = min(3, months)
    recent_reserve_change = results_df['Reward Pool'].iloc[-recent_months:].diff().fillna(0)
    
    recent_burns = recent_reserve_change[recent_reserve_change < 0]
    recent_growth = recent_reserve_change[recent_reserve_change > 0]
    
    recent_monthly_burn = abs(recent_burns.mean()) if not recent_burns.empty else 0
    recent_monthly_growth = recent_growth.mean() if not recent_growth.empty else 0
    
    net_monthly_change = recent_reserve_change.mean()
    
    recent_revenue = results_df['Platform Revenue ($)'].iloc[-recent_months:].mean()
    revenue_burn_ratio = recent_monthly_burn / recent_revenue if recent_revenue > 0 else 0
    
    max_sustainable_from_revenue = recent_revenue * 0.5
    max_sustainable_from_reserve = current_reserve * 0.05
    sustainable_burn = min(max_sustainable_from_revenue, max_sustainable_from_reserve)
    
    is_sustainable = (recent_monthly_burn <= sustainable_burn) if recent_monthly_burn > 0 else True
    
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