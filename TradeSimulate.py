"""
High-Frequency Trading Simulation System
========================================

This module implements a comprehensive trading simulation system that:
1. Imports multiple oscillator strategies
2. Simulates trades with risk management
3. Calculates performance metrics (win rate, profit factor, max drawdown)
4. Provides comparative visualization of strategy performance

Author: Trading Bot Development Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TechnicalIndicators:
    """
    Technical Indicators Class
    Implements various oscillators and indicators for trading strategies
    """
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Formula:
        %K = 100 * (C - L_n) / (H_n - L_n)
        %D = SMA of %K over d_period
        
        Parameters:
        -----------
        high, low, close : pd.Series
            Price data
        k_period : int
            Lookback period for %K calculation
        d_period : int
            Smoothing period for %D calculation
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            %K and %D values
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Parameters:
        -----------
        close : pd.Series
            Closing prices
        period : int
            Calculation period
            
        Returns:
        --------
        pd.Series
            RSI values
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Formula:
        Middle Band = SMA(period)
        Upper Band = Middle Band + (std_dev * σ)
        Lower Band = Middle Band - (std_dev * σ)
        
        Parameters:
        -----------
        close : pd.Series
            Closing prices
        period : int
            Moving average period
        std_dev : float
            Standard deviation multiplier
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            Upper band, middle band (SMA), lower band
        """
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def ema(close: pd.Series, period: int = 50) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Formula:
        EMA_t = α * P_t + (1-α) * EMA_{t-1}
        α = 2 / (period + 1)
        
        Parameters:
        -----------
        close : pd.Series
            Closing prices
        period : int
            EMA period
            
        Returns:
        --------
        pd.Series
            EMA values
        """
        return close.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Formula:
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        
        Parameters:
        -----------
        close : pd.Series
            Closing prices
        fast, slow, signal : int
            MACD parameters
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            MACD line, signal line, histogram
        """
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

class TradingStrategies:
    """
    Trading Strategies Class
    Implements various trading strategies based on technical indicators
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume
        """
        self.data = data.copy()
        self.indicators = TechnicalIndicators()
        
    def stochastic_strategy(self, k_period: int = 14, d_period: int = 3, 
                          overbought: float = 80, oversold: float = 20) -> pd.Series:
        """
        Stochastic Oscillator Strategy
        
        Entry Rules:
        - Long: %K crosses above oversold level and %K > %D
        - Short: %K crosses below overbought level and %K < %D
        
        Parameters:
        -----------
        k_period, d_period : int
            Stochastic parameters
        overbought, oversold : float
            Threshold levels
            
        Returns:
        --------
        pd.Series
            Trading signals (1=Long, -1=Short, 0=No position)
        """
        k_percent, d_percent = self.indicators.stochastic_oscillator(
            self.data['High'], self.data['Low'], self.data['Close'], k_period, d_period
        )
        
        signals = pd.Series(0, index=self.data.index)
        
        # Long signals
        long_condition = (
            (k_percent > oversold) & 
            (k_percent.shift(1) <= oversold) & 
            (k_percent > d_percent)
        )
        
        # Short signals
        short_condition = (
            (k_percent < overbought) & 
            (k_percent.shift(1) >= overbought) & 
            (k_percent < d_percent)
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def rsi_strategy(self, period: int = 14, overbought: float = 70, 
                    oversold: float = 30) -> pd.Series:
        """
        RSI Mean Reversion Strategy
        
        Entry Rules:
        - Long: RSI crosses above oversold level
        - Short: RSI crosses below overbought level
        """
        rsi = self.indicators.rsi(self.data['Close'], period)
        
        signals = pd.Series(0, index=self.data.index)
        
        # Long signals
        long_condition = (rsi > oversold) & (rsi.shift(1) <= oversold)
        
        # Short signals  
        short_condition = (rsi < overbought) & (rsi.shift(1) >= overbought)
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def bollinger_bands_strategy(self, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Bollinger Bands Mean Reversion Strategy
        
        Entry Rules:
        - Long: Price touches lower band
        - Short: Price touches upper band
        """
        upper_band, middle_band, lower_band = self.indicators.bollinger_bands(
            self.data['Close'], period, std_dev
        )
        
        signals = pd.Series(0, index=self.data.index)
        
        # Long signals: Price touches lower band
        long_condition = self.data['Close'] <= lower_band
        
        # Short signals: Price touches upper band
        short_condition = self.data['Close'] >= upper_band
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def ema_crossover_strategy(self, fast_period: int = 50, slow_period: int = 100) -> pd.Series:
        """
        EMA Crossover Strategy
        
        Entry Rules:
        - Long: Fast EMA crosses above Slow EMA
        - Short: Fast EMA crosses below Slow EMA
        """
        ema_fast = self.indicators.ema(self.data['Close'], fast_period)
        ema_slow = self.indicators.ema(self.data['Close'], slow_period)
        
        signals = pd.Series(0, index=self.data.index)
        
        # Long signals: Fast EMA crosses above Slow EMA
        long_condition = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        
        # Short signals: Fast EMA crosses below Slow EMA
        short_condition = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def macd_strategy(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """
        MACD Strategy
        
        Entry Rules:
        - Long: MACD line crosses above signal line
        - Short: MACD line crosses below signal line
        """
        macd_line, signal_line, histogram = self.indicators.macd(
            self.data['Close'], fast, slow, signal
        )
        
        signals = pd.Series(0, index=self.data.index)
        
        # Long signals: MACD crosses above signal
        long_condition = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        
        # Short signals: MACD crosses below signal
        short_condition = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals

class RiskManager:
    """
    Risk Management Class
    Implements various risk management techniques
    """
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_per_trade: float, 
                              stop_loss_pct: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Formula:
        Position Size = (Account Balance * Risk per Trade) / Stop Loss %
        
        Parameters:
        -----------
        account_balance : float
            Current account balance
        risk_per_trade : float
            Risk percentage per trade (e.g., 0.02 for 2%)
        stop_loss_pct : float
            Stop loss percentage (e.g., 0.01 for 1%)
            
        Returns:
        --------
        float
            Position size
        """
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        return min(position_size, account_balance * 0.1)  # Max 10% of account per trade
    
    @staticmethod
    def apply_stop_loss_take_profit(entry_price: float, signal: int, 
                                  stop_loss_pct: float = 0.02, 
                                  take_profit_pct: float = 0.04) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels
        
        Parameters:
        -----------
        entry_price : float
            Entry price
        signal : int
            Trading signal (1=Long, -1=Short)
        stop_loss_pct, take_profit_pct : float
            Risk/reward percentages
            
        Returns:
        --------
        Tuple[float, float]
            Stop loss level, take profit level
        """
        if signal == 1:  # Long position
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # Short position
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
            
        return stop_loss, take_profit

class PerformanceAnalyzer:
    """
    Performance Analysis Class
    Calculates comprehensive trading performance metrics
    """
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, trades: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        trades : pd.DataFrame
            Individual trade records
            
        Returns:
        --------
        Dict
            Performance metrics dictionary
        """
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Trade-based metrics
        if len(trades) > 0:
            win_rate = (trades['pnl'] > 0).mean()
            profit_factor = trades[trades['pnl'] > 0]['pnl'].sum() / abs(trades[trades['pnl'] <= 0]['pnl'].sum()) if trades[trades['pnl'] <= 0]['pnl'].sum() != 0 else np.inf
            avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if (trades['pnl'] > 0).any() else 0
            avg_loss = trades[trades['pnl'] <= 0]['pnl'].mean() if (trades['pnl'] <= 0).any() else 0
            total_trades = len(trades)
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            total_trades = 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'Total Return (%)': total_return * 100,
            'Annual Return (%)': annual_return * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Win Rate (%)': win_rate * 100,
            'Profit Factor': profit_factor,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Total Trades': total_trades,
            'Max Drawdown (%)': max_drawdown * 100
        }

class TradingSimulator:
    """
    Main Trading Simulation Class
    Orchestrates the entire simulation process
    """
    
    def __init__(self, initial_balance: float = 100000, risk_per_trade: float = 0.02):
        """
        Initialize trading simulator
        
        Parameters:
        -----------
        initial_balance : float
            Starting account balance
        risk_per_trade : float
            Risk percentage per trade
        """
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.risk_manager = RiskManager()
        self.performance_analyzer = PerformanceAnalyzer()
        
    def generate_sample_data(self, start_date: str = '2020-01-01', 
                           end_date: str = '2023-12-31', freq: str = 'D') -> pd.DataFrame:
        """
        Generate sample OHLCV data for simulation
        
        Parameters:
        -----------
        start_date, end_date : str
            Date range for data generation
        freq : str
            Data frequency ('D' for daily, 'H' for hourly, etc.)
            
        Returns:
        --------
        pd.DataFrame
            Sample OHLCV data
        """
        np.random.seed(42)  # For reproducible results
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_periods = len(date_range)
        
        # Generate realistic price data using geometric Brownian motion
        initial_price = 100.0
        mu = 0.0001  # Drift
        sigma = 0.02  # Volatility
        
        # Generate random walk
        returns = np.random.normal(mu, sigma, n_periods)
        price_series = initial_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        data = pd.DataFrame(index=date_range)
        data['Close'] = price_series
        
        # Generate realistic OHLC from close prices
        noise = np.random.normal(0, 0.001, n_periods)
        data['Open'] = data['Close'].shift(1).fillna(initial_price) + noise
        
        high_noise = np.abs(np.random.normal(0, 0.005, n_periods))
        low_noise = np.abs(np.random.normal(0, 0.005, n_periods))
        
        data['High'] = np.maximum(data['Open'], data['Close']) + high_noise
        data['Low'] = np.minimum(data['Open'], data['Close']) - low_noise
        
        # Generate volume data
        data['Volume'] = np.random.lognormal(10, 1, n_periods)
        
        return data
    
    def simulate_strategy(self, data: pd.DataFrame, strategy_func, 
                         strategy_name: str, **strategy_params) -> Dict:
        """
        Simulate a trading strategy
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV price data
        strategy_func : callable
            Strategy function to test
        strategy_name : str
            Name of the strategy
        **strategy_params : dict
            Strategy parameters
            
        Returns:
        --------
        Dict
            Simulation results including performance metrics
        """
        print(f"Simulating {strategy_name} strategy...")
        
        # Initialize strategy
        strategy = TradingStrategies(data)
        signals = strategy_func(**strategy_params)
        
        # Initialize tracking variables
        balance = self.initial_balance
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        trades = []
        balance_history = [balance]
        returns = []
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            signal = signals.iloc[i]
            
            # Check for exit conditions if in position
            if position != 0:
                exit_trade = False
                exit_price = current_price
                exit_reason = 'Signal'
                
                # Check stop loss and take profit
                if position == 1:  # Long position
                    if current_price <= stop_loss:
                        exit_trade = True
                        exit_price = stop_loss
                        exit_reason = 'Stop Loss'
                    elif current_price >= take_profit:
                        exit_trade = True
                        exit_price = take_profit
                        exit_reason = 'Take Profit'
                elif position == -1:  # Short position
                    if current_price >= stop_loss:
                        exit_trade = True
                        exit_price = stop_loss
                        exit_reason = 'Stop Loss'
                    elif current_price <= take_profit:
                        exit_trade = True
                        exit_price = take_profit
                        exit_reason = 'Take Profit'
                
                # Check for opposing signal
                if signal != 0 and signal != position:
                    exit_trade = True
                    exit_reason = 'Opposing Signal'
                
                # Execute exit
                if exit_trade:
                    if position == 1:  # Close long
                        pnl = (exit_price - entry_price) / entry_price
                    else:  # Close short
                        pnl = (entry_price - exit_price) / entry_price
                    
                    balance *= (1 + pnl)
                    
                    trades.append({
                        'entry_date': data.index[i-1],
                        'exit_date': data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
            
            # Check for new entry signals
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                stop_loss, take_profit = self.risk_manager.apply_stop_loss_take_profit(
                    entry_price, signal
                )
            
            # Calculate period return
            if len(balance_history) > 0:
                period_return = (balance / balance_history[-1]) - 1
                returns.append(period_return)
            else:
                returns.append(0)
            
            balance_history.append(balance)
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trades)
        returns_series = pd.Series(returns, index=data.index[1:])
        
        # Calculate performance metrics
        metrics = self.performance_analyzer.calculate_metrics(returns_series, trades_df)
        
        return {
            'strategy_name': strategy_name,
            'metrics': metrics,
            'trades': trades_df,
            'returns': returns_series,
            'balance_history': balance_history,
            'final_balance': balance
        }
    
    def run_all_strategies(self, data: pd.DataFrame) -> Dict:
        """
        Run all available strategies on the given data
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV price data
            
        Returns:
        --------
        Dict
            Results for all strategies
        """
        strategy_manager = TradingStrategies(data)
        
        strategies_config = {
            'Stochastic Oscillator': {
                'func': strategy_manager.stochastic_strategy,
                'params': {'k_period': 14, 'd_period': 3, 'overbought': 80, 'oversold': 20}
            },
            'RSI Mean Reversion': {
                'func': strategy_manager.rsi_strategy,
                'params': {'period': 14, 'overbought': 70, 'oversold': 30}
            },
            'Bollinger Bands': {
                'func': strategy_manager.bollinger_bands_strategy,
                'params': {'period': 20, 'std_dev': 2.0}
            },
            'EMA Crossover': {
                'func': strategy_manager.ema_crossover_strategy,
                'params': {'fast_period': 50, 'slow_period': 100}
            },
            'MACD': {
                'func': strategy_manager.macd_strategy,
                'params': {'fast': 12, 'slow': 26, 'signal': 9}
            }
        }
        
        results = {}
        
        for strategy_name, config in strategies_config.items():
            try:
                result = self.simulate_strategy(
                    data, 
                    config['func'], 
                    strategy_name, 
                    **config['params']
                )
                results[strategy_name] = result
            except Exception as e:
                print(f"Error simulating {strategy_name}: {str(e)}")
                continue
        
        return results
    
    def create_performance_visualization(self, results: Dict) -> None:
        """
        Create comprehensive performance visualization
        
        Parameters:
        -----------
        results : Dict
            Results from all strategy simulations
        """
        # Set up the plotting layout
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Cumulative Returns Comparison
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name, result in results.items():
            cumulative_returns = (1 + result['returns']).cumprod()
            ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=strategy_name, linewidth=2)
        
        ax1.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Metrics Comparison
        metrics_df = pd.DataFrame({name: result['metrics'] for name, result in results.items()}).T
        
        # Win Rate
        ax2 = fig.add_subplot(gs[1, 0])
        bars = ax2.bar(metrics_df.index, metrics_df['Win Rate (%)'], color=sns.color_palette("husl", len(metrics_df)))
        ax2.set_title('Win Rate Comparison', fontweight='bold')
        ax2.set_ylabel('Win Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Profit Factor
        ax3 = fig.add_subplot(gs[1, 1])
        bars = ax3.bar(metrics_df.index, metrics_df['Profit Factor'], color=sns.color_palette("husl", len(metrics_df)))
        ax3.set_title('Profit Factor Comparison', fontweight='bold')
        ax3.set_ylabel('Profit Factor')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Maximum Drawdown
        ax4 = fig.add_subplot(gs[1, 2])
        bars = ax4.bar(metrics_df.index, metrics_df['Max Drawdown (%)'], color=sns.color_palette("husl", len(metrics_df)))
        ax4.set_title('Maximum Drawdown Comparison', fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height - 0.5,
                    f'{height:.1f}%', ha='center', va='top')
        
        # 3. Risk-Return Scatter Plot
        ax5 = fig.add_subplot(gs[2, 0])
        for i, (strategy_name, result) in enumerate(results.items()):
            annual_return = result['metrics']['Annual Return (%)']
            volatility = result['metrics']['Volatility (%)']
            ax5.scatter(volatility, annual_return, s=100, 
                       label=strategy_name, alpha=0.7, color=sns.color_palette("husl", len(results))[i])
        
        ax5.set_title('Risk-Return Profile', fontweight='bold')
        ax5.set_xlabel('Volatility (%)')
        ax5.set_ylabel('Annual Return (%)')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 4. Sharpe Ratio Comparison
        ax6 = fig.add_subplot(gs[2, 1])
        bars = ax6.bar(metrics_df.index, metrics_df['Sharpe Ratio'], color=sns.color_palette("husl", len(metrics_df)))
        ax6.set_title('Sharpe Ratio Comparison', fontweight='bold')
        ax6.set_ylabel('Sharpe Ratio')
        ax6.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 5. Total Trades Comparison
        ax7 = fig.add_subplot(gs[2, 2])
        bars = ax7.bar(metrics_df.index, metrics_df['Total Trades'], color=sns.color_palette("husl", len(metrics_df)))
        ax7.set_title('Total Trades Comparison', fontweight='bold')
        ax7.set_ylabel('Number of Trades')
        ax7.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 6. Performance Summary Table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('tight')
        ax8.axis('off')
        
        # Format metrics for display
        display_metrics = metrics_df.round(2)
        
        table = ax8.table(cellText=display_metrics.values,
                         rowLabels=display_metrics.index,
                         colLabels=display_metrics.columns,
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code the cells based on performance
        for i in range(len(display_metrics)):
            for j in range(len(display_metrics.columns)):
                cell = table[(i+1, j)]
                if display_metrics.columns[j] in ['Win Rate (%)', 'Profit Factor', 'Annual Return (%)', 'Sharpe Ratio']:
                    # Higher is better - green gradient
                    cell.set_facecolor('#90EE90' if display_metrics.iloc[i, j] > display_metrics.iloc[:, j].median() else '#FFB6C1')
                elif display_metrics.columns[j] in ['Max Drawdown (%)', 'Volatility (%)']:
                    # Lower is better - reverse color
                    cell.set_facecolor('#FFB6C1' if display_metrics.iloc[i, j] > display_metrics.iloc[:, j].median() else '#90EE90')
        
        plt.suptitle('Trading Strategies Performance Comparison Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary_report(self, results: Dict) -> None:
        """
        Print a comprehensive summary report
        
        Parameters:
        -----------
        results : Dict
            Results from all strategy simulations
        """
        print("\n" + "="*80)
        print("TRADING SIMULATION SUMMARY REPORT")
        print("="*80)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Risk per Trade: {self.risk_per_trade*100:.1f}%")
        print(f"Simulation Period: {len(list(results.values())[0]['returns'])} periods")
        print("\n" + "-"*80)
        
        # Rank strategies by different metrics
        metrics_df = pd.DataFrame({name: result['metrics'] for name, result in results.items()}).T
        
        print("\n🏆 STRATEGY RANKINGS:")
        print("-"*40)
        
        rankings = {
            'Total Return': metrics_df['Total Return (%)'].sort_values(ascending=False),
            'Sharpe Ratio': metrics_df['Sharpe Ratio'].sort_values(ascending=False),
            'Win Rate': metrics_df['Win Rate (%)'].sort_values(ascending=False),
            'Profit Factor': metrics_df['Profit Factor'].sort_values(ascending=False),
            'Min Drawdown': metrics_df['Max Drawdown (%)'].sort_values(ascending=True)
        }
        
        for metric, ranking in rankings.items():
            print(f"\n{metric}:")
            for i, (strategy, value) in enumerate(ranking.head(3).items(), 1):
                if metric == 'Min Drawdown':
                    print(f"  {i}. {strategy}: {value:.2f}%")
                else:
                    print(f"  {i}. {strategy}: {value:.2f}")
        
        # Best overall strategy (simple scoring)
        print("\n" + "-"*40)
        print("🥇 BEST OVERALL STRATEGY:")
        
        # Normalize metrics for scoring (0-1 scale)
        normalized_metrics = metrics_df.copy()
        for col in ['Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Profit Factor']:
            normalized_metrics[col] = (normalized_metrics[col] - normalized_metrics[col].min()) / (normalized_metrics[col].max() - normalized_metrics[col].min())
        
        # Invert drawdown (lower is better)
        normalized_metrics['Max Drawdown (%)'] = 1 - (normalized_metrics['Max Drawdown (%)'] - normalized_metrics['Max Drawdown (%)'].min()) / (normalized_metrics['Max Drawdown (%)'].max() - normalized_metrics['Max Drawdown (%)'].min())
        
        # Calculate composite score
        score_weights = {
            'Total Return (%)': 0.3,
            'Sharpe Ratio': 0.25,
            'Win Rate (%)': 0.2,
            'Profit Factor': 0.15,
            'Max Drawdown (%)': 0.1
        }
        
        composite_scores = pd.Series(0, index=normalized_metrics.index)
        for metric, weight in score_weights.items():
            composite_scores += normalized_metrics[metric] * weight
        
        best_strategy = composite_scores.idxmax()
        print(f"Winner: {best_strategy} (Score: {composite_scores[best_strategy]:.3f})")
        
        # Final balance comparison
        print(f"\n💰 FINAL BALANCE COMPARISON:")
        print("-"*40)
        for strategy_name, result in sorted(results.items(), key=lambda x: x[1]['final_balance'], reverse=True):
            pnl = ((result['final_balance'] / self.initial_balance) - 1) * 100
            print(f"{strategy_name:20}: ${result['final_balance']:10,.2f} ({pnl:+6.2f}%)")
        
        print("\n" + "="*80)

def main():
    """
    Main function to run the trading simulation
    """
    print("🚀 High-Frequency Trading Simulation System")
    print("=" * 50)
    
    # Initialize simulator
    simulator = TradingSimulator(initial_balance=100000, risk_per_trade=0.02)
    
    # Generate sample data
    print("📊 Generating sample market data...")
    data = simulator.generate_sample_data(
        start_date='2020-01-01', 
        end_date='2023-12-31', 
        freq='D'
    )
    print(f"Generated {len(data)} periods of data")
    
    # Run all strategies
    print("\n🔄 Running strategy simulations...")
    results = simulator.run_all_strategies(data)
    
    # Print summary report
    simulator.print_summary_report(results)
    
    # Create visualization
    print("\n📈 Creating performance dashboard...")
    simulator.create_performance_visualization(results)
    
    print("\n✅ Simulation completed successfully!")
    return results

if __name__ == "__main__":
    # Run the simulation
    simulation_results = main()
