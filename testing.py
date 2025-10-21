"""
Testing/Trading Module for Binary Bot
Handles real-time predictions and automated trading

⚠️  WARNING: This module performs live trading with real money!
Use with extreme caution and only with funds you can afford to lose.
"""

import pandas as pd
import numpy as np
import sys
import logging
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import tensorflow as tf

from iq import fast_data, higher, lower, login, get_balance
from training import train_data, add_technical_indicators
from config import (
    SEQ_LEN, FUTURE_PERIOD_PREDICT, DEFAULT_PAIR, DEFAULT_BET_AMOUNT,
    DEFAULT_MARTINGALE, MAX_BET_AMOUNT, MAX_MARTINGALE_STEPS,
    MIN_CONFIDENCE, RETRAIN_INTERVAL, TRADING_SECOND, PREDICTION_SECOND,
    STOCHASTIC_PERIOD, STOCHASTIC_SMOOTH, RSI_PERIOD,
    MA_WINDOWS, EMA_WINDOWS, IQ_BALANCE_MODE, MODEL_DIR, validate_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        logger.warning(f"GPU configuration error: {e}")

def preprocess_prediction(iq, pair: str = DEFAULT_PAIR):
    """
    Preprocess data for prediction

    Args:
        iq: IQ_Option instance
        pair: Trading pair to predict

    Returns:
        Preprocessed array ready for model prediction
    """
    from config import ACTIVE_PAIRS

    logger.debug(f"Preprocessing data for prediction on {pair}")

    try:
        # Fetch data for all active pairs
        main = pd.DataFrame()

        for active in ACTIVE_PAIRS:
            data = fast_data(iq, active)

            if data is None or data.empty:
                logger.warning(f"No data for {active}, skipping")
                continue

            if active == 'EURUSD':
                main = data.drop(columns=['from', 'to'], errors='ignore')
            else:
                current = data.drop(columns=['from', 'to', 'open', 'min', 'max'], errors='ignore')
                current.columns = [f'close_{active}', f'volume_{active}']
                main = main.join(current)

        if main.empty:
            logger.error("No data available for prediction")
            return None

        df = main

        # Clean data - FIXED: Using ffill() instead of fillna(method="ffill")
        df = df.ffill()
        df = df.loc[~df.index.duplicated(keep='first')]

        # Add technical indicators (reusing function from training.py)
        df = add_technical_indicators(df)

        # Drop unnecessary columns
        df = df.drop(columns=['open', 'min', 'max'], errors='ignore')

        # Final cleaning
        df = df.dropna()
        df = df.ffill()
        df = df.dropna()
        df.sort_index(inplace=True)

        # Scale features
        scaler = MinMaxScaler()
        indexes = df.index
        df_scaled = scaler.fit_transform(df)
        pred = pd.DataFrame(df_scaled, index=indexes)

        # Create sequences for LSTM
        sequential_data = []
        prev_days = deque(maxlen=SEQ_LEN)

        for i in pred.iloc[len(pred) - SEQ_LEN:len(pred), :].values:
            prev_days.append([n for n in i[:]])
            if len(prev_days) == SEQ_LEN:
                sequential_data.append([np.array(prev_days)])

        X = []
        for seq in sequential_data:
            X.append(seq)

        result = np.array(X)
        logger.debug(f"Preprocessing complete. Shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Error in preprocess_prediction: {e}", exc_info=True)
        return None

def parse_arguments():
    """Parse command line arguments with validation"""
    if len(sys.argv) == 1:
        return DEFAULT_PAIR, DEFAULT_BET_AMOUNT, DEFAULT_MARTINGALE
    elif len(sys.argv) != 4:
        print("=" * 70)
        print("Binary Bot - Trading Module")
        print("=" * 70)
        print("\nUsage: python testing.py <PAIR> <INITIAL_BET> <MARTINGALE>")
        print("\nArguments:")
        print("  PAIR         - Trading pair (e.g., EURUSD, GBPUSD)")
        print("  INITIAL_BET  - Initial bet amount in dollars (min: 1)")
        print("  MARTINGALE   - Martingale multiplier (default: 2)")
        print("\nExample:")
        print("  python testing.py EURUSD 1 2")
        print("\n⚠️  WARNING: Use PRACTICE mode first!")
        print("=" * 70)
        sys.exit(1)
    else:
        pair = sys.argv[1]
        try:
            bet_money = float(sys.argv[2])
            martingale = float(sys.argv[3])
            if bet_money < 1:
                logger.error("Bet amount must be at least $1")
                sys.exit(1)
            if bet_money > MAX_BET_AMOUNT:
                logger.error(f"Bet amount exceeds maximum: ${MAX_BET_AMOUNT}")
                sys.exit(1)
            if martingale < 1:
                logger.error("Martingale multiplier must be at least 1")
                sys.exit(1)
            return pair, bet_money, martingale
        except ValueError:
            logger.error("Invalid numeric arguments")
            sys.exit(1)


if __name__ == "__main__":
    logger.warning("=" * 70)
    logger.warning("⚠️  BINARY BOT - AUTOMATED TRADING")
    logger.warning("⚠️  Use at your own risk!")
    logger.warning("=" * 70)

    validate_config()
    pair, initial_bet, martingale = parse_arguments()
    logger.info(f"Trading: {pair}, Bet: ${initial_bet}, Martingale: {martingale}x")

    logger.info("Training initial model...")
    model_filename = train_data()
    if not model_filename:
        logger.error("Training failed")
        sys.exit(1)

    model = tf.keras.models.load_model(f"{MODEL_DIR}/{model_filename}.model")
    iq = login()
    if not iq:
        sys.exit(1)

    balance = get_balance(iq)
    logger.info(f"Balance: ${balance}")

    iteration, bet_money, consecutive_losses, bets, result = 0, initial_bet, 0, [], None

    try:
        while True:
            if iteration >= RETRAIN_INTERVAL and iteration % 2 == 0:
                model_filename = train_data()
                if model_filename:
                    model = tf.keras.models.load_model(f"{MODEL_DIR}/{model_filename}.model")
                iteration = 0

            current_second = datetime.datetime.now().second
            if current_second < PREDICTION_SECOND and iteration % 2 == 0:
                pred_ready = preprocess_prediction(iq, pair)
                if pred_ready is None:
                    iteration += 1
                    continue
                pred_ready = pred_ready.reshape(1, SEQ_LEN, pred_ready.shape[3])
                result = model.predict(pred_ready, verbose=0)
                logger.info(f"PUT: {result[0][0]:.4f}, CALL: {result[0][1]:.4f}")
                iteration += 1

            if current_second == TRADING_SECOND and iteration % 2 == 1:
                if result is None:
                    iteration += 1
                    continue

                if bet_money > MAX_BET_AMOUNT:
                    bet_money, consecutive_losses = initial_bet, 0
                if consecutive_losses >= MAX_MARTINGALE_STEPS:
                    bet_money, consecutive_losses = initial_bet, 0

                order_id = None
                if result[0][0] > MIN_CONFIDENCE and result[0][0] > result[0][1]:
                    logger.info(f"PUT ${bet_money}")
                    order_id = lower(iq, bet_money, pair)
                elif result[0][1] > MIN_CONFIDENCE:
                    logger.info(f"CALL ${bet_money}")
                    order_id = higher(iq, bet_money, pair)

                iteration += 1

                if order_id:
                    time.sleep(2)
                    while datetime.datetime.now().second != 1:
                        time.sleep(0.1)
                    try:
                        opts = iq.get_optioninfo_v2(1).get('msg', {}).get('closed_options', [])
                        for opt in opts:
                            if opt.get('win'):
                                bets.append(opt['win'])
                        if bets:
                            last = bets[-1]
                            logger.info(f"Result: {last}")
                            if last == 'win':
                                bet_money, consecutive_losses = initial_bet, 0
                            elif last == 'lose':
                                consecutive_losses += 1
                                bet_money *= martingale
                            logger.info(f"Balance: ${get_balance(iq)}")
                    except Exception as e:
                        logger.error(f"Error: {e}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info(f"\nFinal balance: ${get_balance(iq)}, Trades: {len(bets)}")
        sys.exit(0)
