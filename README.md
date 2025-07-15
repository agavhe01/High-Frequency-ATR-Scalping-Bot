# High-Frequency ATR Scalping Bot

## 🎯 Project Overview

The **High-Frequency ATR Scalping Bot** is a comprehensive algorithmic trading research and simulation platform that implements multiple technical analysis strategies for high-frequency trading. This project combines advanced mathematical indicators with robust risk management systems to evaluate and compare different trading approaches in financial markets.

## 🎪 Project Objectives

### Primary Goals:
1. **Strategy Development**: Implement and test various oscillator-based trading strategies
2. **Risk Management**: Apply sophisticated risk control mechanisms to protect capital
3. **Performance Analysis**: Provide comprehensive metrics for strategy evaluation
4. **Comparative Research**: Enable side-by-side comparison of different trading approaches
5. **Educational Platform**: Serve as a learning tool for algorithmic trading concepts

### Secondary Goals:
- **Scalability**: Design modular architecture for easy strategy addition
- **Reproducibility**: Ensure consistent results through controlled simulation environments
- **Visualization**: Provide clear, professional charts for strategy analysis
- **Documentation**: Maintain comprehensive mathematical and technical documentation

---

## ⚙️ **Setup and Configuration**

### Environment Variables

Several components of this project require environment variables for secure credential management. These variables must be configured before running certain notebooks or scripts.

#### Required Environment Variables

**For Google Sheets Integration** (used in `Indicators/Bollinger_EMA_Indicator.ipynb`):

```bash
# Google Sheets API Credentials
GOOGLE_PROJECT_ID=your-google-cloud-project-id
GOOGLE_PRIVATE_KEY_ID=your-private-key-id-from-service-account
GOOGLE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nyour-complete-private-key\n-----END PRIVATE KEY-----\n"
GOOGLE_CLIENT_EMAIL=your-service-account-email@project-id.iam.gserviceaccount.com
GOOGLE_CLIENT_ID=your-google-client-id
```

#### Setting Environment Variables

**Option 1: Using .env file (Recommended)**
Create a `.env` file in the project root:
```bash
# Copy and paste the variables above with your actual values
# Never commit this file to version control!
```

**Option 2: System Environment Variables**

*Linux/Mac:*
```bash
export GOOGLE_PROJECT_ID="your-project-id"
export GOOGLE_PRIVATE_KEY_ID="your-key-id"
export GOOGLE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nyour-key\n-----END PRIVATE KEY-----"
export GOOGLE_CLIENT_EMAIL="your-email@project.iam.gserviceaccount.com"
export GOOGLE_CLIENT_ID="your-client-id"
```

*Windows:*
```cmd
set GOOGLE_PROJECT_ID=your-project-id
set GOOGLE_PRIVATE_KEY_ID=your-key-id
set GOOGLE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nyour-key\n-----END PRIVATE KEY-----"
set GOOGLE_CLIENT_EMAIL=your-email@project.iam.gserviceaccount.com
set GOOGLE_CLIENT_ID=your-client-id
```

#### Security Best Practices

🔒 **Important Security Notes:**
- Never commit credentials to version control (`.env` files are excluded in `.gitignore`)
- Use service accounts with minimal required permissions
- Regularly rotate API keys and credentials
- Store production credentials in secure key management systems
- Use different credentials for development, testing, and production environments

#### Google Sheets API Setup

To obtain the required credentials:

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Google Sheets API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Sheets API" and enable it

3. **Create Service Account**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Download the JSON key file

4. **Extract Credentials**:
   - Open the downloaded JSON file
   - Extract the required fields for environment variables

5. **Grant Sheet Access**:
   - Share your Google Sheet with the service account email
   - Provide appropriate permissions (read/write as needed)

---

## 📁 File Structure & Component Breakdown

### 📊 **Core Simulation Files**

#### `TradeSimulate.py`
**Purpose**: Main trading simulation engine and orchestration system

**Key Components**:
- **TechnicalIndicators Class**: Implements mathematical calculations for all oscillators
- **TradingStrategies Class**: Defines entry/exit logic for each trading approach
- **RiskManager Class**: Handles position sizing, stop-loss, and take-profit mechanisms
- **PerformanceAnalyzer Class**: Calculates comprehensive trading metrics
- **TradingSimulator Class**: Orchestrates the entire simulation workflow
- **Visualization Engine**: Creates professional performance dashboards

**Functionality**:
- Generates realistic market data using geometric Brownian motion
- Simulates live trading conditions with proper order execution
- Applies real-world constraints like transaction costs and slippage
- Produces detailed performance reports and comparative analysis

**Environment Dependencies**: None (uses simulated data)

---

### 📈 **Strategy Implementation Files**

#### `Scalping Strategy.ipynb`
**Purpose**: Interactive Jupyter notebook implementing EMA and Stochastic Oscillator scalping strategy

**Strategy Logic**:
The scalping strategy combines multiple technical indicators to identify short-term trading opportunities:

**Mathematical Foundation**:
- **Exponential Moving Average (EMA)**:
  ```
  EMA_t = α × P_t + (1-α) × EMA_{t-1}
  where α = 2/(n+1)
  ```

- **Stochastic Oscillator**:
  ```
  %K = 100 × (C - L_n)/(H_n - L_n)
  %D = SMA(%K, 3)
  ```

**Entry Conditions**:
- **Long Signal**: EMA(50) > EMA(100) + %K > 20 + %K < 80 + %K > %D + %K_lag < 20 + Price near EMA(50)
- **Short Signal**: EMA(50) < EMA(100) + %K < %D + %K < 80 + %K > 20 + %K_lag > 80 + Price near EMA(50)

**Risk Management**:
- Transaction costs: 0.5 basis points per trade
- Position sizing based on account balance and volatility
- Dynamic stop-loss and take-profit levels

**Environment Dependencies**: Uses OANDA API configuration file

---

#### `Indicators/Stochastic_indicator.py`
**Purpose**: Standalone implementation of the Stochastic Oscillator

**Mathematical Definition**:
The Stochastic Oscillator is a momentum indicator comparing a security's closing price to its price range over a specific period.

**Formula Breakdown**:
1. **%K Calculation** (Fast Stochastic):
   ```
   %K = 100 × (Current Close - Lowest Low)/(Highest High - Lowest Low)
   ```
   Where:
   - Current Close = Most recent closing price
   - Lowest Low = Lowest price in the lookback period
   - Highest High = Highest price in the lookback period

2. **%D Calculation** (Slow Stochastic):
   ```
   %D = Simple Moving Average of %K over 3 periods
   ```

**Trading Interpretation**:
- **Overbought**: %K > 80 (potential sell signal)
- **Oversold**: %K < 20 (potential buy signal)
- **Crossover Signals**: %K crossing above/below %D
- **Divergence**: Price and oscillator moving in opposite directions

**Parameters**:
- **Lookback Period**: Typically 14 periods for %K calculation
- **Smoothing Period**: Usually 3 periods for %D calculation
- **Overbought/Oversold Levels**: Commonly 80/20 or 70/30

**Environment Dependencies**: None (standalone mathematical implementation)

---

#### `Indicators/Bollinger_EMA_Indicator.ipynb`
**Purpose**: Enhanced Bollinger Bands implementation using Exponential Moving Average

**Mathematical Framework**:

**Traditional Bollinger Bands**:
```
Middle Band = SMA(Close, n)
Upper Band = Middle Band + (k × σ)
Lower Band = Middle Band - (k × σ)
```

**Enhanced EMA Version**:
```
Middle Band = EMA(Typical Price, n)
Upper Band = EMA + (k × σ)
Lower Band = EMA - (k × σ)
```

Where:
- **Typical Price** = (Open + High + Low + Close) / 4
- **EMA** = Exponential Moving Average with α = 2/(n+1)
- **σ** = Standard deviation of typical price over n periods
- **k** = Multiplier (typically 1.7 or 2.0)
- **n** = Lookback period (typically 20)

**Trading Applications**:

1. **Mean Reversion Strategy**:
   - **Buy Signal**: Price touches or breaks below lower band
   - **Sell Signal**: Price touches or breaks above upper band
   - **Exit**: Price returns to middle band (EMA)

2. **Volatility Breakout Strategy**:
   - **Squeeze**: Bands contract (low volatility period)
   - **Expansion**: Bands widen (high volatility period)
   - **Breakout**: Price moves outside bands with volume confirmation

3. **Trend Following Enhancement**:
   - **Uptrend**: Price consistently above middle band
   - **Downtrend**: Price consistently below middle band
   - **Sideways**: Price oscillates between bands

**Advantages of EMA-Based Approach**:
- More responsive to recent price changes
- Faster signal generation for short-term strategies
- Better suited for high-frequency trading applications
- Reduced lag compared to Simple Moving Average

**Environment Dependencies**: **Requires Google Sheets API credentials** (see Environment Variables section above)

---

### 🛡️ **Risk Management Documentation**

#### Position Sizing Algorithm
**Formula**:
```
Position Size = (Account Balance × Risk per Trade) / Stop Loss %
Maximum Position = min(Position Size, Account Balance × 0.10)
```

**Risk Parameters**:
- **Risk per Trade**: 2% of account balance
- **Stop Loss**: 2% from entry price
- **Take Profit**: 4% from entry price (2:1 reward-to-risk ratio)
- **Maximum Position**: 10% of total account balance

#### Stop-Loss and Take-Profit Calculations

**For Long Positions**:
```
Stop Loss = Entry Price × (1 - Stop Loss %)
Take Profit = Entry Price × (1 + Take Profit %)
```

**For Short Positions**:
```
Stop Loss = Entry Price × (1 + Stop Loss %)
Take Profit = Entry Price × (1 - Take Profit %)
```

---

### 📊 **Performance Metrics Framework**

#### Core Performance Indicators

1. **Win Rate**:
   ```
   Win Rate = (Number of Profitable Trades / Total Trades) × 100
   ```

2. **Profit Factor**:
   ```
   Profit Factor = Gross Profit / Gross Loss
   ```
   - Values > 1.0 indicate profitable strategies
   - Values > 1.5 considered good
   - Values > 2.0 considered excellent

3. **Maximum Drawdown**:
   ```
   Max Drawdown = (Peak Value - Trough Value) / Peak Value × 100
   ```

4. **Sharpe Ratio**:
   ```
   Sharpe Ratio = (Average Return - Risk-Free Rate) / Standard Deviation
   ```
   - Measures risk-adjusted returns
   - Values > 1.0 considered good
   - Values > 2.0 considered excellent

5. **Annual Return**:
   ```
   Annual Return = (Ending Value / Beginning Value)^(252/n) - 1
   ```
   Where n = number of trading periods

#### Advanced Metrics

6. **Sortino Ratio**:
   ```
   Sortino Ratio = (Average Return - Risk-Free Rate) / Downside Deviation
   ```

7. **Calmar Ratio**:
   ```
   Calmar Ratio = Annual Return / Maximum Drawdown
   ```

8. **Average Trade Duration**:
   ```
   Avg Duration = Total Holding Time / Number of Trades
   ```

---

### 🔧 **Configuration and Setup Files**

#### `.gitignore`
**Purpose**: Defines files and directories to exclude from version control

**Key Exclusions**:
- **Credentials**: API keys, configuration files, authentication tokens
- **Data Files**: CSV files, datasets, historical data
- **Environment**: Virtual environments, compiled files, cache
- **IDE Files**: Editor-specific configuration and temporary files
- **Log Files**: Trading logs, error logs, debug information
- **OS Files**: System-generated files (.DS_Store, Thumbs.db)

**Security Benefits**:
- Prevents accidental exposure of sensitive trading credentials
- Keeps repository clean and focused on source code
- Protects proprietary data and trading algorithms

---

## 🔬 **Technical Analysis Strategies Implemented**

### 1. **Stochastic Oscillator Strategy**
**Mathematical Basis**: Momentum oscillator comparing closing price to price range
**Signal Generation**: Crossover of %K and %D lines combined with overbought/oversold levels
**Best For**: Range-bound markets, mean reversion scenarios
**Time Frame**: Short to medium-term (minutes to hours)

### 2. **RSI Mean Reversion Strategy**
**Mathematical Basis**: Relative Strength Index measuring price change velocity
**Signal Generation**: Oversold (RSI < 30) buy signals, Overbought (RSI > 70) sell signals
**Best For**: Choppy, sideways markets with clear support/resistance levels
**Time Frame**: Medium-term (hours to days)

### 3. **Bollinger Bands Strategy**
**Mathematical Basis**: Volatility bands around moving average
**Signal Generation**: Price touching bands for mean reversion, band width for volatility
**Best For**: Both trending and ranging markets, volatility analysis
**Time Frame**: Flexible (minutes to weeks)

### 4. **EMA Crossover Strategy**
**Mathematical Basis**: Exponential moving average crossovers
**Signal Generation**: Fast EMA crossing above/below slow EMA
**Best For**: Trending markets with clear directional bias
**Time Frame**: Medium to long-term (hours to days)

### 5. **MACD Strategy**
**Mathematical Basis**: Moving Average Convergence Divergence
**Signal Generation**: MACD line crossing signal line, histogram analysis
**Best For**: Trend confirmation and momentum analysis
**Time Frame**: Medium-term (hours to days)

---

## 📈 **Expected Performance Characteristics**

### Strategy Comparison Matrix

| Strategy | Expected Win Rate | Profit Factor | Max Drawdown | Best Market |
|----------|------------------|---------------|---------------|-------------|
| Stochastic | 45-60% | 1.2-1.8 | 15-25% | Ranging |
| RSI | 50-65% | 1.3-2.0 | 12-20% | Sideways |
| Bollinger Bands | 40-55% | 1.1-1.6 | 18-28% | All Markets |
| EMA Crossover | 35-50% | 1.5-2.5 | 20-35% | Trending |
| MACD | 40-55% | 1.2-1.9 | 16-26% | Trending |

### Risk-Return Profiles

**Conservative Strategies** (Lower risk, moderate returns):
- RSI Mean Reversion
- Bollinger Bands

**Aggressive Strategies** (Higher risk, potentially higher returns):
- EMA Crossover
- MACD Momentum

**Balanced Strategies** (Moderate risk and returns):
- Stochastic Oscillator

---

## 🎓 **Educational Value**

### Learning Outcomes
1. **Technical Analysis**: Understanding mathematical foundations of trading indicators
2. **Risk Management**: Implementing proper position sizing and risk controls
3. **Strategy Development**: Creating systematic trading approaches
4. **Performance Evaluation**: Analyzing trading results with statistical rigor
5. **Market Dynamics**: Understanding different market conditions and strategy effectiveness

### Research Applications
- **Academic Studies**: Backtesting efficiency of technical indicators
- **Strategy Optimization**: Parameter tuning and performance enhancement
- **Market Analysis**: Understanding indicator behavior across different market conditions
- **Risk Assessment**: Evaluating downside protection mechanisms

---

## 🚀 **Future Enhancements**

### Planned Features
1. **Additional Indicators**: Williams %R, Commodity Channel Index, Average True Range
2. **Machine Learning Integration**: ML-based signal confirmation and filtering
3. **Multi-Timeframe Analysis**: Combining signals across different time horizons
4. **Real-Time Data Integration**: Connection to live market data feeds
5. **Portfolio Management**: Multi-asset strategy allocation and correlation analysis

### Advanced Risk Management
1. **Value at Risk (VaR)**: Statistical risk measurement
2. **Kelly Criterion**: Optimal position sizing based on edge and odds
3. **Correlation Analysis**: Cross-asset risk assessment
4. **Dynamic Stop-Loss**: Adaptive risk management based on volatility

---

## 📚 **References and Further Reading**

### Technical Analysis Literature
- "Technical Analysis of the Financial Markets" by John J. Murphy
- "New Concepts in Technical Trading Systems" by J. Welles Wilder Jr.
- "Bollinger on Bollinger Bands" by John Bollinger

### Quantitative Trading Resources
- "Algorithmic Trading" by Ernest P. Chan
- "Building Winning Algorithmic Trading Systems" by Kevin Davey

### Risk Management Studies
- "The Mathematics of Money Management" by Ralph Vince

---

**Note**: This project is designed for educational and research purposes. All strategies should be thoroughly tested with paper trading before considering live implementation. Past performance does not guarantee future results, and all trading involves risk of loss. 