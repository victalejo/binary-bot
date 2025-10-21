# -*- coding: utf-8 -*-
"""
IQ Option API Interface Module
Handles connection, data retrieval, and trading operations
"""

import logging
import time
import sys
from typing import Optional, List, Dict, Any
from iqoptionapi.stable_api import IQ_Option
import pandas as pd

# Import configuration
try:
    from config import (
        IQ_USERNAME, IQ_PASSWORD, IQ_BALANCE_MODE,
        ACTIVE_PAIRS, CANDLE_SIZE, CANDLES_COUNT, FAST_DATA_COUNT,
        validate_config
    )
except ImportError:
    print("ERROR: config.py not found. Please ensure config.py exists.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def login(verbose: bool = False) -> Optional[IQ_Option]:
    """
    Login to IQ Option with error handling

    Args:
        verbose: Enable debug logging

    Returns:
        IQ_Option instance or None if login fails
    """
    # Validate configuration
    config_status = validate_config()
    if not config_status.get('credentials', False):
        logger.error("Missing credentials. Set IQ_USERNAME and IQ_PASSWORD environment variables.")
        return None

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    try:
        logger.info(f"Attempting login to IQ Option as {IQ_USERNAME}")
        iq = IQ_Option(IQ_USERNAME, IQ_PASSWORD)

        # Check connection
        check, reason = iq.connect()
        if not check:
            logger.error(f"Connection failed: {reason}")
            return None

        logger.info("Successfully connected to IQ Option")

        # Change balance mode
        iq.change_balance(IQ_BALANCE_MODE)
        logger.info(f"Balance mode set to: {IQ_BALANCE_MODE}")

        if IQ_BALANCE_MODE == "REAL":
            logger.warning("⚠️  TRADING WITH REAL MONEY! Exercise caution!")

        return iq

    except Exception as e:
        logger.error(f"Login failed with exception: {e}", exc_info=True)
        return None


def higher(iq: IQ_Option, money: float, active: str, duration: int = 1) -> Optional[int]:
    """
    Place a CALL (buy) order

    Args:
        iq: IQ_Option instance
        money: Amount to bet
        active: Trading pair (e.g., 'EURUSD')
        duration: Duration in minutes

    Returns:
        Order ID or None if failed
    """
    try:
        logger.info(f"Placing CALL order: {active}, Amount: ${money}, Duration: {duration}m")
        done, order_id = iq.buy(money, active, "call", duration)

        if not done:
            logger.error(f"CALL order failed for {active}")
            return None

        logger.info(f"CALL order placed successfully. ID: {order_id}")
        return order_id

    except Exception as e:
        logger.error(f"Exception placing CALL order: {e}", exc_info=True)
        return None


def lower(iq: IQ_Option, money: float, active: str, duration: int = 1) -> Optional[int]:
    """
    Place a PUT (sell) order

    Args:
        iq: IQ_Option instance
        money: Amount to bet
        active: Trading pair (e.g., 'EURUSD')
        duration: Duration in minutes

    Returns:
        Order ID or None if failed
    """
    try:
        logger.info(f"Placing PUT order: {active}, Amount: ${money}, Duration: {duration}m")
        done, order_id = iq.buy(money, active, "put", duration)

        if not done:
            logger.error(f"PUT order failed for {active}")
            return None

        logger.info(f"PUT order placed successfully. ID: {order_id}")
        return order_id

    except Exception as e:
        logger.error(f"Exception placing PUT order: {e}", exc_info=True)
        return None
  
def get_candles(iq: IQ_Option, active: str, size: int = None, count: int = None,
                end_time: float = None) -> Optional[List[Dict]]:
    """
    Get candlestick data for a trading pair

    Args:
        iq: IQ_Option instance
        active: Trading pair
        size: Candle size in seconds (default from config)
        count: Number of candles (default from config)
        end_time: End time timestamp (default: now)

    Returns:
        List of candle dictionaries or None if failed
    """
    size = size or CANDLE_SIZE
    count = count or CANDLES_COUNT
    end_time = end_time or time.time()

    try:
        logger.debug(f"Fetching {count} candles for {active}")
        candles = iq.get_candles(active, size, count, end_time)

        if not candles:
            logger.warning(f"No candles returned for {active}")
            return None

        logger.debug(f"Retrieved {len(candles)} candles for {active}")
        return candles

    except Exception as e:
        logger.error(f"Error fetching candles for {active}: {e}", exc_info=True)
        return None


def get_all_candles(iq: IQ_Option, active: str, start_candle: float,
                    batches: int = 1) -> List[Dict]:
    """
    Get multiple batches of historical candles

    Args:
        iq: IQ_Option instance
        active: Trading pair
        start_candle: Starting timestamp
        batches: Number of batches to fetch

    Returns:
        List of all candles
    """
    final_data = []

    try:
        for batch in range(batches):
            logger.debug(f"Fetching batch {batch + 1}/{batches} for {active}")
            data = iq.get_candles(active, CANDLE_SIZE, CANDLES_COUNT, start_candle)

            if data and len(data) > 0:
                start_candle = data[0]['to'] - 1
                final_data.extend(data)
            else:
                logger.warning(f"No data in batch {batch + 1} for {active}")
                break

        logger.info(f"Retrieved {len(final_data)} total candles for {active}")
        return final_data

    except Exception as e:
        logger.error(f"Error in get_all_candles for {active}: {e}", exc_info=True)
        return final_data

def get_data_needed(iq: IQ_Option) -> Optional[pd.DataFrame]:
    """
    Gather all required data from multiple trading pairs

    Args:
        iq: IQ_Option instance

    Returns:
        DataFrame with combined data or None if failed
    """
    try:
        start_candle = time.time()
        actives = ACTIVE_PAIRS
        final_data = pd.DataFrame()

        for active in actives:
            logger.info(f"Fetching data for {active}")
            current = get_all_candles(iq, active, start_candle)

            if not current:
                logger.warning(f"No data retrieved for {active}, skipping")
                continue

            # Convert candles to DataFrame - FIXED: Using concat instead of append
            frames = []
            for candle in current:
                useful_frame = pd.DataFrame(
                    list(candle.values()),
                    index=list(candle.keys())
                ).T.drop(columns=['at'])
                useful_frame = useful_frame.set_index(useful_frame['id']).drop(columns=['id'])
                frames.append(useful_frame)

            if frames:
                main = pd.concat(frames, ignore_index=False)
                main = main.drop_duplicates()

                if active == 'EURUSD':
                    final_data = main.drop(columns={'from', 'to'})
                else:
                    main = main.drop(columns={'from', 'to', 'open', 'min', 'max'})
                    main.columns = [f'close_{active}', f'volume_{active}']
                    final_data = final_data.join(main)

        final_data = final_data.loc[~final_data.index.duplicated(keep='first')]
        logger.info(f"Data gathering complete. Shape: {final_data.shape}")
        return final_data

    except Exception as e:
        logger.error(f"Error in get_data_needed: {e}", exc_info=True)
        return None


def fast_data(iq: IQ_Option, pair: str) -> Optional[pd.DataFrame]:
    """
    Gather reduced data for quick predictions

    Args:
        iq: IQ_Option instance
        pair: Trading pair (e.g., 'EURUSD')

    Returns:
        DataFrame with recent candles or None if failed
    """
    try:
        logger.debug(f"Fetching fast data for {pair}")
        candles = iq.get_candles(pair, CANDLE_SIZE, FAST_DATA_COUNT, time.time())

        if not candles:
            logger.warning(f"No fast data retrieved for {pair}")
            return None

        # Convert to DataFrame - FIXED: Using concat instead of append
        frames = []
        for candle in candles:
            useful_frame = pd.DataFrame(
                list(candle.values()),
                index=list(candle.keys())
            ).T.drop(columns=['at'])
            useful_frame = useful_frame.set_index(useful_frame['id']).drop(columns=['id'])
            frames.append(useful_frame)

        if frames:
            main = pd.concat(frames, ignore_index=False)
            logger.debug(f"Fast data retrieved. Shape: {main.shape}")
            return main
        else:
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error in fast_data for {pair}: {e}", exc_info=True)
        return None


def get_balance(iq: IQ_Option) -> Optional[float]:
    """
    Get current account balance

    Args:
        iq: IQ_Option instance

    Returns:
        Balance or None if failed
    """
    try:
        balance = iq.get_balance()
        logger.info(f"Current balance: ${balance}")
        return balance
    except Exception as e:
        logger.error(f"Error getting balance: {e}", exc_info=True)
        return None


def get_profit(iq: IQ_Option, pair: str = 'EURUSD') -> Optional[float]:
    """
    Get profit percentage for a trading pair

    Args:
        iq: IQ_Option instance
        pair: Trading pair

    Returns:
        Profit percentage or None if failed
    """
    try:
        all_profit = iq.get_all_profit()
        profit = all_profit.get(pair, {}).get('turbo')
        logger.debug(f"Profit for {pair}: {profit}%")
        return profit
    except Exception as e:
        logger.error(f"Error getting profit for {pair}: {e}", exc_info=True)
        return None
