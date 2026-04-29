def calculate_rti(df, length=5):
    """
    Range Tightening Index (RTI) - Pine Script v6 conversion
    RTI < 15% = Low volatility contraction (buying condition)
    """
    # Volatility = daily range
    df['volatility'] = df['High'] - df['Low']
    
    # Max/min volatility over lookback
    df['max_vol'] = df['volatility'].rolling(window=length).max()
    df['min_vol'] = df['volatility'].rolling(window=length).min()
    
    # RTI normalized 0-100
    df['rti'] = 100 * (df['volatility'] - df['min_vol']) / (df['max_vol'] - df['min_vol'])
    
    return df

def rti_buy_constraint(df, rti_threshold=15):
    """
    PPO Strategy Constraint: RTI must be < 15% on buying day OR previous EOD
    """
    # Check current OR previous day RTI
    current_rti_low = df['rti'].iloc[-1] < rti_threshold
    prev_rti_low = df['rti'].iloc[-2] < rti_threshold
    
    return current_rti_low or prev_rti_low

# PPO Environment Integration
class PPOTadingEnv(gym.Env):
    def __init__(self, df):
        self.df = calculate_rti(df.copy())
        self.current_step = 0
        
    def step(self, action):
        # RTI Constraint (MUST PASS for BUY)
        rti_condition = rti_buy_constraint(self.df.iloc[:self.current_step+1])
        
        if action == 1:  # BUY action
            if not rti_condition:
                action = 0  # Force HOLD
                reward_penalty = -0.1  # Penalty for invalid entry
        
        # Execute trade logic...
        return next_state, reward, done, info

# Usage Example
spx_data = yf.download('SPY', period='1y')
spx_data = calculate_rti(spx_data)

# Check today's RTI condition
today_rti_ok = rti_buy_constraint(spx_data.tail(2))
print(f"RTI Buy Condition Met: {today_rti_ok}")
print(f"Current RTI: {spx_data['rti'].iloc[-1]:.1f}%")