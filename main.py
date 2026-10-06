import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_synthetic_market_regimes(S0, mu, sigma, T=5.0, steps_per_year=252, num_simulations=1000, seed=42):
    """Generates synthetic price paths using Geometric Brownian Motion (GBM)."""
    np.random.seed(seed)
    total_steps = int(T * steps_per_year)
    dt = 1.0 / steps_per_year
    
    Z = np.random.standard_normal((num_simulations, total_steps))
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    log_returns = drift + diffusion
    cumulative_returns = np.cumsum(log_returns, axis=1)
    
    S0_column = np.full((num_simulations, 1), S0)
    price_paths = S0 * np.exp(cumulative_returns)
    return np.hstack((S0_column, price_paths))


def compute_adx_and_atr(paths_df, period=14):
    """
    Computes path-wise ATR and simplified ADX (Trend Strength) across all paths simultaneously.
    """
    high = paths_df
    low = paths_df
    close = paths_df
    
    # 1. True Range (TR)
    prev_close = close.shift(1).fillna(close)
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = tr.rolling(window=period).mean()
    
    # 2. Simplified Directional Movement & ADX
    price_diff = close.diff().fillna(0.0)
    pos_dm = np.where(price_diff > 0, price_diff, 0.0)
    neg_dm = np.where(price_diff < 0, -price_diff, 0.0)
    
    pos_di = 100 * pd.DataFrame(pos_dm).rolling(period).mean() / atr
    neg_di = 100 * pd.DataFrame(neg_dm).rolling(period).mean() / atr
    
    dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di + 1e-8))
    adx = dx.rolling(window=period).mean().fillna(0.0)
    
    return atr.fillna(0.0).values, adx.values


def apply_dynamic_atr_stop(paths_matrix, raw_positions, atr_matrix, atr_mult=2.0):
    """
    Applies dynamic ATR-based trailing stop-loss across all paths simultaneously.
    """
    num_steps, num_paths = paths_matrix.shape
    final_positions = np.zeros_like(raw_positions)
    
    in_position = np.zeros(num_paths, dtype=bool)
    peak_prices = np.zeros(num_paths)
    stop_prices = np.zeros(num_paths)
    
    for t in range(num_steps):
        raw_pos = raw_positions[t, :]
        prices = paths_matrix[t, :]
        atrs = atr_matrix[t, :]
        
        # 1. Entry: Signal Long (1) while out of position
        entry = (raw_pos == 1.0) & (~in_position)
        in_position[entry] = True
        peak_prices[entry] = prices[entry]
        stop_prices[entry] = prices[entry] - (atr_mult * atrs[entry])
        
        # 2. Update Peak Prices & Dynamic ATR Stop for Active Positions
        holding = in_position & (raw_pos == 1.0)
        new_peaks = np.maximum(peak_prices[holding], prices[holding])
        peak_prices[holding] = new_peaks
        stop_prices[holding] = np.maximum(stop_prices[holding], new_peaks - (atr_mult * atrs[holding]))
        
        # 3. Dynamic Stop Trigger
        stopped_out = holding & (prices < stop_prices)
        in_position[stopped_out] = False
        peak_prices[stopped_out] = 0.0
        stop_prices[stopped_out] = 0.0
        
        # 4. Normal Signal Exit
        exited = in_position & (raw_pos == 0.0)
        in_position[exited] = False
        peak_prices[exited] = 0.0
        stop_prices[exited] = 0.0
        
        final_positions[t, :] = np.where(in_position, 1.0, 0.0)
        
    return final_positions


if __name__ == "__main__":
    # --- STAGE 1: 5-Year Monte Carlo Path Generation ---
    paths = generate_synthetic_market_regimes(S0=100.0, mu=0.08, sigma=0.18, T=5.0, num_simulations=1000)
    paths_df = pd.DataFrame(paths.T)

    # --- STAGE 2: Upgraded Signals (12/26 EMA + ADX Trend Strength Filter) ---
    ema_fast = paths_df.ewm(span=12, adjust=False).mean()
    ema_slow = paths_df.ewm(span=26, adjust=False).mean()
    
    atr_matrix, adx_matrix = compute_adx_and_atr(paths_df, period=14)
    
    # Condition: EMA Bullish Crossover AND ADX > 20 (Strong Trend Only)
    raw_signals = np.where((ema_fast > ema_slow) & (adx_matrix > 20.0), 1.0, 0.0)
    
    # 1-Day Execution Lag (No Lookahead Bias)
    raw_positions = np.roll(raw_signals, shift=1, axis=0)
    raw_positions[0, :] = 0.0

    # Dynamic ATR Trailing Stop-Loss Execution
    final_positions = apply_dynamic_atr_stop(paths_df.values, raw_positions, atr_matrix, atr_mult=2.0)

    # --- STAGE 3: Compounding Portfolio Returns & Cash Yield ---
    r_annual = 0.04
    r_daily = (1.0 + r_annual) ** (1.0 / 252.0) - 1.0

    asset_returns = paths_df.pct_change().fillna(0.0).values
    strategy_returns = (final_positions * asset_returns) + ((1.0 - final_positions) * r_daily)
    equity_curves = np.cumprod(1.0 + strategy_returns, axis=0)

    # --- STAGE 4: Risk & Performance Metrics ---
    mean_daily = np.mean(strategy_returns, axis=0)
    std_daily = np.std(strategy_returns, axis=0)
    std_daily = np.where(std_daily == 0, np.nan, std_daily)

    sharpe_ratios = (mean_daily / std_daily) * np.sqrt(252)
    clean_sharpes = sharpe_ratios[~np.isnan(sharpe_ratios)]

    running_peaks = np.maximum.accumulate(equity_curves, axis=0)
    drawdowns = (equity_curves - running_peaks) / running_peaks
    max_drawdowns = np.min(drawdowns, axis=0)

    print(f"=== PROJECT 3 UPGRADED STRATEGY RESULTS ===")
    print(f"Mean Sharpe Ratio across 1,000 regimes: {np.mean(clean_sharpes):.2f}")
    print(f"Worst-Case Max Drawdown across 1,000 regimes: {np.min(max_drawdowns) * 100:.2f}%")

    # --- STAGE 5: Visualizations ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    np.random.seed(42)
    sample_indices = np.random.choice(1000, size=100, replace=False)
    for idx in sample_indices:
        axes[0].plot(equity_curves[:, idx], color="gray", alpha=0.15, linewidth=0.8)
    
    median_path = np.median(equity_curves, axis=1)
    axes[0].plot(median_path, color="blue", linewidth=2, label="Median Equity Curve")
    axes[0].set_title("Project 3: Upgraded Strategy Equity Curves (1,000 Paths)")
    axes[0].set_xlabel("Trading Days")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(clean_sharpes, bins=40, color="lightgreen", edgecolor="black", alpha=0.7)
    mean_s = np.mean(clean_sharpes)
    axes[1].axvline(mean_s, color="red", linestyle="--", linewidth=2, label=f"Mean Sharpe: {mean_s:.2f}")
    axes[1].set_title("Out-of-Sample Sharpe Ratio Distribution")
    axes[1].set_xlabel("Sharpe Ratio")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("stress_test_results.png", dpi=300)
    plt.show()