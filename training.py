"""
Training Module for Binary Bot
Handles data preprocessing, feature engineering, and LSTM model training
"""

import numpy as np
import pandas as pd
import random
import logging
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import time

# Import from local modules
from iq import get_data_needed, login
from config import (
    SEQ_LEN, FUTURE_PERIOD_PREDICT, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    LSTM_UNITS, DROPOUT_RATES, DENSE_UNITS, VALIDATION_SPLIT,
    MA_WINDOWS, EMA_WINDOWS, RSI_PERIOD, STOCHASTIC_PERIOD, STOCHASTIC_SMOOTH,
    LOG_DIR, MODEL_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure GPU settings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        logger.warning(f"GPU configuration error: {e}")

def classify(current: float, future: float) -> int:
    """
    Classify price movement: 1 for up, 0 for down

    Args:
        current: Current price
        future: Future price

    Returns:
        1 if price goes up, 0 otherwise
    """
    return 1 if float(future) > float(current) else 0


def preprocess_df(df: pd.DataFrame) -> tuple:
    """
    Preprocess dataframe for LSTM training

    Args:
        df: DataFrame with features and target

    Returns:
        Tuple of (X, y) arrays for training
    """
    logger.debug(f"Preprocessing dataframe. Shape: {df.shape}")

    # Drop future column
    df = df.drop("future", axis=1)

    # Scale features
    scaler = MinMaxScaler()
    indexes = df.index
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, index=indexes)
    
    # Create sequences
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])  # Store all features except target
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)
    logger.debug(f"Created {len(sequential_data)} sequences")

    # Separate by class for balanced dataset
    buys = []   # CALL orders (price up)
    sells = []  # PUT orders (price down)

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    # Balance dataset by taking equal samples from each class
    lower = min(len(buys), len(sells))
    logger.info(f"Balancing dataset: {len(buys)} buys, {len(sells)} sells -> using {lower} each")

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    # Split into features and targets
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    logger.debug(f"Final dataset: X shape {np.array(X).shape}, y length {len(y)}")
    return np.array(X), y  



def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to dataframe

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    logger.info("Calculating technical indicators...")

    # Moving averages
    df['MA_20'] = df['close'].rolling(window=MA_WINDOWS[0]).mean()
    df['MA_50'] = df['close'].rolling(window=MA_WINDOWS[1]).mean()

    # Exponential moving averages
    df['EMA_20'] = df['close'].ewm(span=EMA_WINDOWS[0], adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=EMA_WINDOWS[1], adjust=False).mean()

    # Stochastic oscillator
    df['L14'] = df['min'].rolling(window=STOCHASTIC_PERIOD).min()
    df['H14'] = df['max'].rolling(window=STOCHASTIC_PERIOD).max()
    df['%K'] = 100 * ((df['close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=STOCHASTIC_SMOOTH).mean()

    # RSI (Relative Strength Index)
    chg = df['close'].diff(1)
    gain = chg.mask(chg < 0, 0)
    loss = chg.mask(chg > 0, 0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = abs(avg_gain / avg_loss)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Clean up temporary columns
    df = df.drop(columns=['L14', 'H14'], errors='ignore')

    logger.info("Technical indicators calculated")
    return df


def train_data() -> str:
    """
    Main training function: fetch data, engineer features, train model

    Returns:
        Model filename
    """
    logger.info("=" * 70)
    logger.info("Starting training process")
    logger.info("=" * 70)

    # Login to IQ Option
    iq = login()
    if not iq:
        logger.error("Login failed. Cannot proceed with training.")
        return None

    # Fetch data
    logger.info("Fetching market data...")
    df = get_data_needed(iq)

    if df is None or df.empty:
        logger.error("No data retrieved. Cannot train model.")
        return None

    logger.info(f"Data shape: {df.shape}")

    # Clean data
    logger.info("Cleaning data...")
    df = df.ffill()  # FIXED: Using ffill() instead of fillna(method="ffill")
    df = df.loc[~df.index.duplicated(keep='first')]

    # Add future price for target
    df['future'] = df["close"].shift(-FUTURE_PERIOD_PREDICT)

    # Add technical indicators
    df = add_technical_indicators(df)

    # Drop highly correlated or intermediate columns
    df = df.drop(columns=['open', 'min', 'max'], errors='ignore')

    # Final cleaning
    df = df.dropna()
    df = df.ffill()
    df = df.dropna()
    df.sort_index(inplace=True)

    # Create target variable
    logger.info("Creating target variable...")
    df['target'] = list(map(classify, df['close'], df['future']))
    df.dropna(inplace=True)

    target_counts = df['target'].value_counts()
    logger.info(f"Target distribution: {dict(target_counts)}")

    # Convert to float32 for efficiency
    df = df.astype('float32')

    # Split into train and validation sets
    logger.info(f"Splitting data with {VALIDATION_SPLIT*100}% validation...")
    times = sorted(df.index.values)
    split_idx = -int(VALIDATION_SPLIT * len(times))
    last_train_time = times[split_idx]

    validation_df = df[(df.index >= last_train_time)]
    train_df = df[(df.index < last_train_time)]

    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(validation_df)} samples")

    # Preprocess data
    train_x, train_y = preprocess_df(train_df)
    validation_x, validation_y = preprocess_df(validation_df)

    logger.info(f"Train data: {len(train_x)} sequences, sells: {train_y.count(0)}, buys: {train_y.count(1)}")
    logger.info(f"Validation data: {len(validation_x)} sequences, sells: {validation_y.count(0)}, buys: {validation_y.count(1)}")

    train_y = np.asarray(train_y)
    validation_y = np.asarray(validation_y)

    # Model naming
    NAME = f"{LEARNING_RATE}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-{EPOCHS}-{BATCH_SIZE}-PRED-{int(time.time())}"
    logger.info(f"Model name: {NAME}")
    # Create model directories if they don't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Build LSTM model
    logger.info("Building LSTM model...")
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(LSTM_UNITS[0], input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(DROPOUT_RATES[0]))
    model.add(BatchNormalization())

    # Second LSTM layer
    model.add(LSTM(LSTM_UNITS[1], return_sequences=True))
    model.add(Dropout(DROPOUT_RATES[1]))
    model.add(BatchNormalization())

    # Third LSTM layer
    model.add(LSTM(LSTM_UNITS[2]))
    model.add(Dropout(DROPOUT_RATES[2]))
    model.add(BatchNormalization())

    # Dense layers
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(Dropout(0.2))

    # Output layer (2 classes: PUT/CALL)
    model.add(Dense(2, activation='softmax'))

    # FIXED: Updated Adam optimizer API (lr -> learning_rate)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=5e-5)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)

    # Callbacks
    tensorboard = TensorBoard(log_dir=f"{LOG_DIR}/{NAME}")

    filepath = "LSTM-best"
    # FIXED: val_acc -> val_accuracy in newer Keras versions
    checkpoint = ModelCheckpoint(
        f"{MODEL_DIR}/{filepath}.model",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train model
    logger.info("=" * 70)
    logger.info("Starting model training...")
    logger.info("=" * 70)

    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=[tensorboard, checkpoint, early_stopping],
        verbose=1
    )

    # Log training results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    logger.info("=" * 70)
    logger.info(f"Training complete!")
    logger.info(f"Final training accuracy: {final_train_acc:.4f}")
    logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
    logger.info(f"Model saved as: {filepath}")
    logger.info("=" * 70)

    return filepath

