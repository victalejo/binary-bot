# -*- coding: utf-8 -*-
"""
Configuration file for binary-bot
Use environment variables for sensitive data
"""

import os
from typing import List, Dict

# ============================================================================
# API CREDENTIALS - NEVER HARDCODE THESE
# ============================================================================
# Set these as environment variables:
# export IQ_USERNAME="your_email@example.com"
# export IQ_PASSWORD="your_password"
# ============================================================================

IQ_USERNAME = os.getenv('IQ_USERNAME')
IQ_PASSWORD = os.getenv('IQ_PASSWORD')
IQ_BALANCE_MODE = os.getenv('IQ_BALANCE_MODE', 'PRACTICE')  # PRACTICE or REAL

# ============================================================================
# TRADING PARAMETERS
# ============================================================================

# Trading pairs to analyze
ACTIVE_PAIRS: List[str] = ['EURUSD', 'GBPUSD', 'EURJPY', 'AUDUSD']
DEFAULT_PAIR: str = 'EURUSD'

# Money management
DEFAULT_BET_AMOUNT: float = 1.0  # Minimum bet in dollars
MAX_BET_AMOUNT: float = 100.0    # Maximum allowed bet (safety limit)
DEFAULT_MARTINGALE: float = 2.0  # Martingale multiplier
MAX_MARTINGALE_STEPS: int = 5    # Maximum consecutive losses before reset

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

SEQ_LEN: int = 5                 # Sequence length for LSTM
FUTURE_PERIOD_PREDICT: int = 2   # Prediction horizon
LEARNING_RATE: float = 0.001
EPOCHS: int = 40
BATCH_SIZE: int = 16

# Model architecture
LSTM_UNITS: List[int] = [128, 128, 128]
DROPOUT_RATES: List[float] = [0.2, 0.1, 0.2]
DENSE_UNITS: int = 32

# ============================================================================
# TECHNICAL INDICATORS PARAMETERS
# ============================================================================

MA_WINDOWS: List[int] = [20, 50]
EMA_WINDOWS: List[int] = [20, 50]
RSI_PERIOD: int = 14
STOCHASTIC_PERIOD: int = 14
STOCHASTIC_SMOOTH: int = 3

# ============================================================================
# DATA PARAMETERS
# ============================================================================

CANDLE_SIZE: int = 60            # Candle size in seconds
CANDLES_COUNT: int = 1000        # Number of candles to fetch
FAST_DATA_COUNT: int = 300       # Candles for quick predictions
VALIDATION_SPLIT: float = 0.1    # 10% for validation

# ============================================================================
# TRADING LOGIC PARAMETERS
# ============================================================================

MIN_CONFIDENCE: float = 0.5      # Minimum confidence to trade
RETRAIN_INTERVAL: int = 10       # Retrain model every N predictions
TRADING_SECOND: int = 59         # Second of the minute to place bet
PREDICTION_SECOND: int = 30      # Start prediction before this second

# ============================================================================
# LOGGING AND MONITORING
# ============================================================================

LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR: str = 'logs'
MODEL_DIR: str = 'models'

# ============================================================================
# SAFETY CHECKS
# ============================================================================

def validate_config() -> Dict[str, bool]:
    """
    Validate configuration and return status
    """
    issues = {}

    if not IQ_USERNAME or not IQ_PASSWORD:
        issues['credentials'] = False
        print("⚠️  WARNING: IQ_USERNAME and IQ_PASSWORD environment variables not set!")
        print("   Set them with: export IQ_USERNAME='your_email' export IQ_PASSWORD='your_password'")
    else:
        issues['credentials'] = True

    if IQ_BALANCE_MODE not in ['PRACTICE', 'REAL']:
        issues['balance_mode'] = False
        print(f"⚠️  WARNING: Invalid balance mode '{IQ_BALANCE_MODE}'. Using PRACTICE.")
    else:
        issues['balance_mode'] = True

    if IQ_BALANCE_MODE == 'REAL':
        print("\n" + "="*70)
        print("⚠️  CRITICAL WARNING: REAL MONEY MODE ENABLED!")
        print("   This bot will trade with REAL MONEY. Use at your own risk!")
        print("   Consider using PRACTICE mode first.")
        print("="*70 + "\n")

    if DEFAULT_BET_AMOUNT < 1.0:
        issues['bet_amount'] = False
        print(f"⚠️  WARNING: Bet amount {DEFAULT_BET_AMOUNT} is below minimum (1.0)")
    else:
        issues['bet_amount'] = True

    return issues

if __name__ == '__main__':
    print("Binary Bot Configuration")
    print("="*50)
    print(f"Balance Mode: {IQ_BALANCE_MODE}")
    print(f"Active Pairs: {ACTIVE_PAIRS}")
    print(f"Default Bet: ${DEFAULT_BET_AMOUNT}")
    print(f"Martingale: {DEFAULT_MARTINGALE}x")
    print(f"Model: LSTM {LSTM_UNITS}")
    print("="*50)
    validate_config()
