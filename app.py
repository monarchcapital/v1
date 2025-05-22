import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, date, datetime
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Monarch: Stock Price Predictor", layout="wide")

# --- Global list for collecting training messages ---
training_messages_log = []

# Define the expected columns for the prediction log globally
PREDICTION_LOG_COLUMNS = ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used', 'ticker', 'model_used', 'predicted_value', 'actual_close']

# --- Prediction Logging Functions ---
def save_prediction(ticker, prediction_for_date, predicted_value, actual_close_price, model_name, prediction_generation_date, training_end_date_used):
    """
    Saves a single prediction entry to a CSV log file.
    Includes logic to prevent duplicate entries based on date, ticker, model, and generation date.
    """
    predictions_dir = "monarch_predictions_data"
    os.makedirs(predictions_dir, exist_ok=True)
    file_path = os.path.join(predictions_dir, f"{ticker}_predictions_log.csv")

    new_data_dict = {
        'prediction_generation_date': prediction_generation_date.strftime("%Y-%m-%d %H:%M:%S"),
        'prediction_for_date': prediction_for_date.strftime("%Y-%m-%d"),
        'ticker': ticker,
        'model_used': model_name,
        'predicted_value': predicted_value,
        'actual_close': actual_close_price if actual_close_price is not None else np.nan
    }
    # Ensure training_end_date_used is formatted correctly
    if isinstance(training_end_date_used, (date, datetime)):
        new_data_dict['training_end_date_used'] = training_end_date_used.strftime("%Y-%m-%d")
    else:
        new_data_dict['training_end_date_used'] = str(training_end_date_used)

    new_data = pd.DataFrame([new_data_dict])
    new_data = new_data[PREDICTION_LOG_COLUMNS] # Ensure consistent column order

    try:
        existing_df = pd.DataFrame(columns=PREDICTION_LOG_COLUMNS) # Default to empty
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                # Robust read for duplicate check: expect all 7 columns, skip header
                existing_df = pd.read_csv(
                    file_path,
                    names=PREDICTION_LOG_COLUMNS,
                    header=None,
                    skiprows=1, # Skip the actual header line
                    dtype=str    # Read all as string initially
                )
                # Coerce types needed for duplicate check
                existing_df['prediction_generation_date'] = pd.to_datetime(existing_df['prediction_generation_date'], errors='coerce')
                existing_df['prediction_for_date'] = pd.to_datetime(existing_df['prediction_for_date'], errors='coerce')
                existing_df['training_end_date_used'] = pd.to_datetime(existing_df['training_end_date_used'], errors='coerce') # Ensure this is also datetime
                # Ensure 'ticker' and 'model_used' are strings for comparison
                existing_df['ticker'] = existing_df['ticker'].astype(str)
                existing_df['model_used'] = existing_df['model_used'].astype(str)

            except pd.errors.EmptyDataError: # Can happen if skiprows=1 on a 1-line file (only header)
                existing_df = pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)
            except Exception as e_read:
                st.warning(f"Could not reliably read existing log {file_path} for duplicate check: {e_read}. This might lead to duplicate entries if the log is corrupted.")
                existing_df = pd.DataFrame(columns=PREDICTION_LOG_COLUMNS) # Proceed assuming no duplicates can be checked

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            new_data.to_csv(file_path, index=False, header=True) # Write header for new file
        else:
            # Duplicate check logic
            is_duplicate = False
            if not existing_df.empty:
                 # Ensure date components are correctly extracted for comparison
                check_gen_date = prediction_generation_date.date() # prediction_generation_date is datetime.datetime
                check_for_date = prediction_for_date # prediction_for_date is already datetime.date
                check_training_end_date = training_end_date_used # training_end_date_used is already datetime.date

                mask = (
                    (existing_df['prediction_generation_date'].dt.date == check_gen_date) &
                    (existing_df['prediction_for_date'].dt.date == check_for_date) &
                    (existing_df['ticker'] == ticker) &
                    (existing_df['model_used'] == model_name) &
                    # IMPORTANT: Include training_end_date_used in duplicate check
                    (existing_df['training_end_date_used'].dt.date == check_training_end_date)
                )
                is_duplicate = mask.any()

            if not is_duplicate:
                new_data.to_csv(file_path, mode='a', header=False, index=False)
            # else:
                # st.info(f"Duplicate prediction for {ticker} on {prediction_for_date.strftime('%Y-%m-%d')} by {model_name} generated on {prediction_generation_date.strftime('%Y-%m-%d')} not saved.")

    except Exception as e:
        st.error(f"Error saving prediction for {ticker} on {prediction_for_date.strftime('%Y-%m-%d')}: {e}")

def load_past_predictions(ticker):
    """
    Loads past predictions for a given ticker from its CSV log file.
    Handles empty or corrupted files gracefully.
    """
    predictions_dir = "monarch_predictions_data"
    file_path = os.path.join(predictions_dir, f"{ticker}_predictions_log.csv")
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            # Robust read: expect all 7 columns, skip header
            df = pd.read_csv(
                file_path,
                names=PREDICTION_LOG_COLUMNS,
                header=None,
                skiprows=1, # Skip the actual header line
                dtype=str    # Read all as string initially
            )

            # Convert to specific types
            df['prediction_generation_date'] = pd.to_datetime(df['prediction_generation_date'], errors='coerce')
            df['prediction_for_date'] = pd.to_datetime(df['prediction_for_date'], errors='coerce')
            df['training_end_date_used'] = pd.to_datetime(df['training_end_date_used'], errors='coerce')
            df['predicted_value'] = pd.to_numeric(df['predicted_value'], errors='coerce')
            df['actual_close'] = pd.to_numeric(df['actual_close'], errors='coerce')
            
            # Ensure 'ticker' and 'model_used' are strings
            df['ticker'] = df['ticker'].astype(str)
            df['model_used'] = df['model_used'].astype(str) # Corrected from model to model_used

            return df
        except pd.errors.EmptyDataError: # File had only a header
            return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)
        except Exception as e:
            st.error(f"Error loading past predictions for {ticker} from {file_path}: {e}. File might be corrupted.")
            return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)
    return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)

def get_previous_trading_day(current_date):
    """
    Calculates the previous trading day (skipping weekends).
    Args:
        current_date (datetime.date or pandas.Timestamp): The date for which to find the previous trading day.
    Returns:
        datetime.date: The previous trading day.
    """
    # Convert to datetime.date if it's a pandas.Timestamp
    if isinstance(current_date, pd.Timestamp):
        current_date = current_date.date()

    prev_day = current_date - timedelta(days=1)
    # Loop backwards until a weekday is found
    while prev_day.weekday() >= 5: # Saturday (5) or Sunday (6)
        prev_day -= timedelta(days=1)
    return prev_day


# --- UI Elements ---
st.title("👑 Monarch: Your Stock Price Oracle")
st.markdown("""
Welcome to **Monarch**, a sophisticated stock price prediction platform.
This application employs a suite of advanced machine learning models and technical indicators
to analyze historical market data and forecast potential future price movements.
Leverage Monarch's analytical capabilities to gain deeper insights into stock trends,
evaluate model performance through backtesting, and explore future price projections.
""")

# Sidebar inputs
st.sidebar.header("🛠️ Configuration Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, use .NS for Indian stocks):", value="AAPL").upper()

today = date.today()
default_end_bt = today - timedelta(days=1)
default_start_bt = default_end_bt - timedelta(days=3*365) # Default to 3 years for more data

st.sidebar.subheader("🗓️ Training Period")
start_bt = st.sidebar.date_input("Training Start Date (t1):", value=default_start_bt, help="Start date for model training data.")
end_bt = st.sidebar.date_input("Training End Date (t2):", value=default_end_bt, help="End date for model training data. Predictions will start from t2+1.")

if start_bt >= end_bt:
    st.sidebar.error("Training Start Date (t1) must be before Training End Date (t2)")
    st.stop()

st.sidebar.subheader("🤖 Model Selection")
model_choices = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Linear Regression', 'SVR', 'KNN', 'Decision Tree']
model_choice = st.sidebar.selectbox("Select Main Model (for Close Price):", model_choices, help="The primary model used for predictions.")
perform_tuning = st.sidebar.checkbox("Perform Hyperparameter Tuning", value=False, help="May significantly increase training time but can improve model accuracy.")
n_future = st.sidebar.slider("Predict Future Days (after t2):", min_value=1, max_value=90, value=15, help="Number of future trading days to forecast.")

st.sidebar.subheader("📊 Model Comparison")
compare_models = st.sidebar.multiselect("Select Models to Compare:", model_choices, default=model_choices[:3], help="Additional models to compare against the main model on recent data.")
train_days_comparison = st.sidebar.slider("Recent Data for Comparison (days):", min_value=30, max_value=1000, value=180, step=10, help="How many recent days of data to use for the model comparison chart.")

st.sidebar.subheader("⚙️ Technical Indicator Settings")
st.sidebar.markdown("---")
st.sidebar.markdown("**Moving Average (MA):** *Common: 10, 20, 50, 200 days.*")
ma_input = st.sidebar.text_input("MA Windows (comma-separated):", value="10,20,50")
try:
    ma_windows_list = [int(x.strip()) for x in ma_input.split(',') if x.strip()]
    if not ma_windows_list: ma_windows_list = [10, 20, 50]
    if any(w <= 0 for w in ma_windows_list):
         st.sidebar.error("MA windows must be positive.")
         st.stop()
except ValueError:
    st.sidebar.error("Invalid MA input.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Volatility (Std Dev):** *Common: 10, 20 days.*")
std_input = st.sidebar.text_input("Volatility Windows (comma-separated):", value="10,20")
try:
    std_windows_list = [int(x.strip()) for x in std_input.split(',') if x.strip()]
    if not std_windows_list: std_windows_list = [10, 20]
    if any(w <= 0 for w in std_windows_list):
         st.sidebar.error("Volatility windows must be positive.")
         st.stop()
except ValueError:
    st.sidebar.error("Invalid Volatility input.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Relative Strength Index (RSI):** *Common: 14 days.*")
rsi_window = st.sidebar.number_input("RSI Window:", min_value=1, value=14)

st.sidebar.markdown("---")
st.sidebar.markdown("**MACD:** *Common: 12, 26, 9 days.*")
macd_short_window = st.sidebar.number_input("MACD Short Window:", min_value=1, value=12)
macd_long_window = st.sidebar.number_input("MACD Long Window:", min_value=1, value=26)
macd_signal_window = st.sidebar.number_input("MACD Signal Window:", min_value=1, value=9)

st.sidebar.markdown("---")
st.sidebar.markdown("**Bollinger Bands (BB):** *Common: 20 day window, 2.0 std dev.*")
bb_window = st.sidebar.number_input("BB Window:", min_value=1, value=20)
bb_std_dev = st.sidebar.number_input("BB Std Dev Multiplier:", min_value=0.1, value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Average True Range (ATR):** *Common: 14 days.*")
atr_window = st.sidebar.number_input("ATR Window:", min_value=1, value=14)

st.sidebar.markdown("---")
st.sidebar.markdown("**Stochastic Oscillator:** *Common: 14 day %K, 3 day %D.*")
stoch_window = st.sidebar.number_input("Stochastic %K Window:", min_value=1, value=14)
stoch_smooth_window = st.sidebar.number_input("Stochastic %D Window:", min_value=1, value=3)
st.sidebar.markdown("---")

lag_features_list = [1, 2, 3, 5, 10] # Fixed lag features

# --- Training Log Display Area ---
log_expander = st.sidebar.expander("📜 Training Log & Messages", expanded=False)
log_placeholder = log_expander.empty()


# --- Data Loading & Feature Engineering ---
@st.cache_data(show_spinner="Fetching market data...", ttl=timedelta(hours=1))
def download_data(ticker_symbol):
    """
    Downloads historical stock data for a given ticker symbol using yfinance.
    Performs initial data cleaning and renames columns for consistency.
    """
    try:
        data_df = yf.download(ticker_symbol, period="max", progress=False)
        if data_df.empty:
            st.error(f"No data for {ticker_symbol}. Check symbol or try later.")
            return pd.DataFrame()
        
        data_df.reset_index(inplace=True)
        
        # Robust column cleaning
        cleaned_columns = []
        for col in data_df.columns:
            if isinstance(col, tuple):
                # If it's a tuple (potential MultiIndex), take the first element
                cleaned_col = str(col[0]).lower().strip()
            else:
                cleaned_col = str(col).lower().strip()
            cleaned_columns.append(cleaned_col)
        data_df.columns = cleaned_columns
        
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in data_df.columns]
             st.error(f"Data for {ticker_symbol} missing required columns after cleaning: {missing}. Available columns: {data_df.columns.tolist()}")
             return pd.DataFrame()
             
        data_df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        # Ensure only OHLCV and Date are kept, in correct order
        data_df = data_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df.dropna(inplace=True)
        return data_df
    except Exception as e:
        st.error(f"Error downloading or processing data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def create_features(df, lags, ma_wins, std_wins, rsi_w, macd_s, macd_l, macd_sig, bb_w, bb_std, atr_w, stoch_k, stoch_d):
    """
    Creates various technical indicator features and lag features from the raw stock data.
    """
    df_feat = df.copy()
    df_feat.columns = df_feat.columns.astype(str)
    df_feat['Day'] = np.arange(len(df_feat))

    # Lag features
    for lag in lags:
        df_feat[f'Close_Lag_{lag}'] = df_feat['Close'].shift(lag)
        df_feat[f'Open_Lag_{lag}'] = df_feat['Open'].shift(lag)
    
    # Moving Averages
    for win in ma_wins: df_feat[f'MA_{win}'] = df_feat['Close'].rolling(window=win).mean()
    
    # Volatility (Standard Deviation)
    for win in std_wins: df_feat[f'Volatility_{win}'] = df_feat['Close'].rolling(window=win).std()

    # On-Balance Volume (OBV)
    df_feat['Price_Change'] = df_feat['Close'].diff()
    df_feat['Volume_Direction'] = 0
    df_feat.loc[df_feat['Price_Change'] > 0, 'Volume_Direction'] = df_feat['Volume']
    df_feat.loc[df_feat['Price_Change'] < 0, 'Volume_Direction'] = -df_feat['Volume']
    df_feat['OBV'] = df_feat['Volume_Direction'].cumsum()

    # Relative Strength Index (RSI)
    delta = df_feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(com=rsi_w-1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_w-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_feat['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df_feat['Close'].ewm(span=macd_s, adjust=False).mean()
    exp2 = df_feat['Close'].ewm(span=macd_l, adjust=False).mean()
    df_feat['MACD'] = exp1 - exp2
    df_feat['MACD_Signal'] = df_feat['MACD'].ewm(span=macd_sig, adjust=False).mean()
    df_feat['MACD_Hist'] = df_feat['MACD'] - df_feat['MACD_Signal']

    # Bollinger Bands (BB)
    df_feat['BB_Middle'] = df_feat['Close'].rolling(window=bb_w).mean()
    rolling_std = df_feat['Close'].rolling(window=bb_w).std()
    std_multiplier_series = pd.Series((rolling_std.values * bb_std).ravel(), index=df_feat.index)
    df_feat['BB_Upper'] = df_feat['BB_Middle'] + std_multiplier_series
    df_feat['BB_Lower'] = df_feat['BB_Middle'] - std_multiplier_series

    # Average True Range (ATR)
    df_feat['High_Low'] = df_feat['High'] - df_feat['Low']
    df_feat['High_PrevClose'] = np.abs(df_feat['High'] - df_feat['Close'].shift(1))
    df_feat['Low_PrevClose'] = np.abs(df_feat['Low'] - df_feat['Close'].shift(1))
    df_feat['True_Range'] = df_feat[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    df_feat['ATR'] = df_feat['True_Range'].ewm(span=atr_w, adjust=False).mean()

    # Stochastic Oscillator
    lowest_low = df_feat['Low'].rolling(window=stoch_k).min()
    highest_high = df_feat['High'].rolling(window=stoch_k).max()
    df_feat['%K'] = ((df_feat['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)) * 100
    df_feat['%D'] = df_feat['%K'].rolling(window=stoch_d).mean()

    # Drop intermediate columns used for feature calculation
    df_feat.drop(columns=['Price_Change', 'Volume_Direction', 'High_Low', 'High_PrevClose', 'Low_PrevClose', 'True_Range'], errors='ignore', inplace=True)
    
    # Drop rows with NaN values introduced by feature creation (e.g., due to rolling windows or shifts)
    cols_to_keep_for_iterative = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols_only = [col for col in df_feat.columns if col not in cols_to_keep_for_iterative]
    df_feat_cleaned = df_feat.dropna(subset=feature_cols_only).copy()
    return df_feat_cleaned

# --- Model Training & Prediction ---
def get_model(name):
    """Returns an instance of the specified machine learning model."""
    if name == 'Random Forest': return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if name == 'Linear Regression': return LinearRegression(n_jobs=-1)
    if name == 'SVR': return SVR(kernel='rbf', C=100, gamma=0.1) # Epsilon default is fine
    if name == 'XGBoost': return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, random_state=42, early_stopping_rounds=10, eval_metric='rmse', tree_method='hist', verbosity=0, n_jobs=-1)
    if name == 'Gradient Boosting': return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    if name == 'KNN': return KNeighborsRegressor(n_neighbors=7, n_jobs=-1) # Tuned n_neighbors slightly
    if name == 'Decision Tree': return DecisionTreeRegressor(random_state=42, max_depth=10)
    return LinearRegression(n_jobs=-1) # Default fallback

def _train_single_model(X_scaled, y, model_name, perform_tuning_flag, display_name):
    """
    Trains a single model, optionally performing hyperparameter tuning.
    Logs messages to a global list.
    """
    global training_messages_log # Use global list
    base_model = get_model(model_name)
    trained_model = None
    best_params = None

    if perform_tuning_flag:
        training_messages_log.append(f"⏳ Tuning {display_name}...")
        param_dist = {
            'Random Forest': {'n_estimators': randint(50, 250), 'max_depth': [None] + list(randint(5, 25).rvs(10)), 'min_samples_split': randint(2, 11), 'min_samples_leaf': randint(1, 11)},
            'SVR': {'C': uniform(1, 1000), 'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(10)), 'epsilon': uniform(0.01, 0.5)},
            'XGBoost': {'n_estimators': randint(100, 700), 'learning_rate': uniform(0.01, 0.15), 'max_depth': randint(3, 12), 'subsample': uniform(0.6, 0.4), 'colsample_bytree': uniform(0.6, 0.4)}, # subsample + colsample_bytree should sum to <=1, fixed to be 0.6 to 1.0
            'Gradient Boosting': {'n_estimators': randint(50, 250), 'learning_rate': uniform(0.01, 0.15), 'max_depth': randint(3, 10), 'min_samples_split': randint(2, 11), 'min_samples_leaf': randint(1, 11)},
            'KNN': {'n_neighbors': randint(3, 25)},
            'Decision Tree': {'max_depth': [None] + list(randint(3, 20).rvs(5)), 'min_samples_split': randint(2, 11), 'min_samples_leaf': randint(1, 11)}
        }

        if model_name in param_dist:
            n_iter, cv_folds = 15, 3 # Reduced for speed
            if len(X_scaled) < cv_folds * 2: # Ensure enough samples for CV
                training_messages_log.append(f"⚠️ Not enough data for {cv_folds}-fold CV for {display_name} ({len(X_scaled)} samples). Skipping tuning.")
                perform_tuning_flag = False
            
            if perform_tuning_flag:
                random_search = RandomizedSearchCV(base_model, param_distributions=param_dist[model_name],
                                                   n_iter=n_iter, scoring='neg_mean_squared_error', cv=cv_folds,
                                                   random_state=42, n_jobs=-1, verbose=0)
                try:
                    random_search.fit(X_scaled, y)
                    trained_model = random_search.best_estimator_
                    best_params = random_search.best_params_
                    training_messages_log.append(f"✅ Best params for {display_name}: {best_params}")
                except Exception as e:
                    training_messages_log.append(f"❌ Error tuning {display_name}: {e}. Using defaults.")
                    perform_tuning_flag = False # Fallback to default
        else:
            training_messages_log.append(f"ℹ️ Tuning not configured for {display_name}. Using defaults.")
            perform_tuning_flag = False

    if not perform_tuning_flag: # Train with defaults if tuning skipped or failed
        training_messages_log.append(f"⚙️ Training {display_name} with defaults...")
        try:
            if model_name == 'XGBoost':
                 train_size = int(len(X_scaled) * 0.8)
                 if len(X_scaled) - train_size > 10: # Ensure validation set is meaningful
                     X_train_split, X_val_split = X_scaled[:train_size], X_scaled[train_size:]
                     y_train_split, y_val_split = y[:train_size], y[train_size:]
                     base_model.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], verbose=False)
                 else:
                     base_model.fit(X_scaled, y) # Not enough for val set
                 trained_model = base_model
            else:
                base_model.fit(X_scaled, y)
                trained_model = base_model
            training_messages_log.append(f"👍 {display_name} trained successfully with defaults.")
        except Exception as e:
            training_messages_log.append(f"❌ Error training {display_name} with defaults: {e}")
            trained_model = None
            
    return trained_model, best_params

def train_models_pipeline(df_train, model_choice_main, perform_tuning_main):
    """
    Trains models for Close, Open, and Volatility targets using the specified model type.
    Returns a dictionary of trained models, scalers, and features.
    """
    global training_messages_log
    if df_train.empty:
        training_messages_log.append("❌ Training data is empty. Cannot train models.")
        return None

    base_feature_cols = [col for col in df_train.columns if col not in ['Date']]
    
    # Define features, ensuring no target leakage for each model
    # Features for Close price prediction should not include future prices or volatility related to future prices
    feature_cols_close = [col for col in base_feature_cols if col not in ['Close', 'Open', 'High', 'Low', 'Volume'] + [f'Volatility_{w}' for w in std_windows_list]]
    # Features for Open price prediction
    feature_cols_open = [col for col in base_feature_cols if col not in ['Open', 'Close', 'High', 'Low', 'Volume'] + [f'Volatility_{w}' for w in std_windows_list]]
    volatility_target_col_name = f'Volatility_{std_windows_list[0]}' if std_windows_list else None
    # Features for Volatility prediction
    feature_cols_volatility = [col for col in base_feature_cols if col not in [f'Volatility_{w}' for w in std_windows_list] + ['Close', 'Open', 'High', 'Low', 'Volume']]

    models_data = {}

    # Close Price Model
    if not feature_cols_close: training_messages_log.append("❌ No features for Close model.")
    else:
        X_close = df_train[feature_cols_close].copy()
        y_close = df_train['Close'].values.ravel()
        if len(X_close) != len(y_close) or len(X_close) == 0: training_messages_log.append(f"❌ Data mismatch for Close model ({len(X_close)} vs {len(y_close)}).")
        else:
            scaler_close = StandardScaler()
            X_close_scaled = scaler_close.fit_transform(X_close)
            model_c, params_c = _train_single_model(X_close_scaled, y_close, model_choice_main, perform_tuning_main, f"{model_choice_main} (Close)")
            models_data['Close'] = {'model': model_c, 'scaler': scaler_close, 'features': feature_cols_close, 'params': params_c}

    # Open Price Model (using same model type as Close for simplicity, can be changed)
    if not feature_cols_open: training_messages_log.append("❌ No features for Open model.")
    else:
        X_open = df_train[feature_cols_open].copy()
        y_open = df_train['Open'].values.ravel()
        if len(X_open) != len(y_open) or len(X_open) == 0: training_messages_log.append(f"❌ Data mismatch for Open model ({len(X_open)} vs {len(y_open)}).")
        else:
            scaler_open = StandardScaler()
            X_open_scaled = scaler_open.fit_transform(X_open)
            model_o, params_o = _train_single_model(X_open_scaled, y_open, model_choice_main, perform_tuning_main, f"{model_choice_main} (Open)")
            models_data['Open'] = {'model': model_o, 'scaler': scaler_open, 'features': feature_cols_open, 'params': params_o}

    # Volatility Model
    if volatility_target_col_name and volatility_target_col_name in df_train.columns:
        if not feature_cols_volatility: training_messages_log.append("❌ No features for Volatility model.")
        else:
            X_vol = df_train[feature_cols_volatility].copy()
            y_vol = df_train[volatility_target_col_name].values.ravel()
            if len(X_vol) != len(y_vol) or len(X_vol) == 0: training_messages_log.append(f"❌ Data mismatch for Volatility model ({len(X_vol)} vs {len(y_vol)}).")
            else:
                scaler_vol = StandardScaler()
                X_vol_scaled = scaler_vol.fit_transform(X_vol)
                model_v, params_v = _train_single_model(X_vol_scaled, y_vol, model_choice_main, perform_tuning_main, f"{model_choice_main} (Volatility)")
                models_data['Volatility'] = {'model': model_v, 'scaler': scaler_vol, 'features': feature_cols_volatility, 'target_col': volatility_target_col_name, 'params': params_v}
    else:
        training_messages_log.append(f"ℹ️ Volatility target '{volatility_target_col_name}' not found. Skipping Volatility model.")
        
    return models_data

def generate_predictions_pipeline(df_data, models_info_dict):
    """
    Generates predictions for various targets using the provided trained models.
    """
    predictions_output = {}
    if df_data.empty or not models_info_dict:
        return predictions_output

    for target_name, info in models_info_dict.items():
        model = info.get('model')
        scaler = info.get('scaler')
        feature_cols = info.get('features')
        
        if not all([model, scaler, feature_cols]):
            # training_messages_log.append(f"⚠️ Skipping {target_name} prediction: missing model/scaler/features.") # Redundant if logged during training
            predictions_output[target_name] = pd.DataFrame()
            continue

        df_data.columns = df_data.columns.astype(str)
        X_pred_df = df_data.reindex(columns=feature_cols, fill_value=0).copy()

        if X_pred_df.empty:
            # training_messages_log.append(f"⚠️ No features for {target_name} prediction after reindex.")
            predictions_output[target_name] = pd.DataFrame()
            continue
        
        expected_num_features = len(feature_cols)
        if X_pred_df.shape[1] != expected_num_features:
            # training_messages_log.append(f"⚠️ Feature mismatch for {target_name}: expected {expected_num_features}, got {X_pred_df.shape[1]}.")
            predictions_output[target_name] = pd.DataFrame()
            continue
            
        try:
            X_pred_scaled = scaler.transform(X_pred_df)
            predicted_values = model.predict(X_pred_scaled)
        except Exception as e:
            # training_messages_log.append(f"❌ Error predicting {target_name}: {e}")
            predictions_output[target_name] = pd.DataFrame()
            continue

        result_df = pd.DataFrame({'Date': df_data['Date'].values, f'Predicted {target_name}': predicted_values.ravel()})
        
        actual_col = None
        if target_name == 'Close' and 'Close' in df_data: actual_col = 'Close'
        elif target_name == 'Open' and 'Open' in df_data: actual_col = 'Open'
        elif target_name == 'Volatility' and info.get('target_col') in df_data: actual_col = info.get('target_col')

        if actual_col:
            result_df[f'Actual {target_name}'] = df_data[actual_col].values.ravel()
            result_df['Difference'] = result_df[f'Actual {target_name}'] - result_df[f'Predicted {target_name}']
            for col in [f'Actual {target_name}', f'Predicted {target_name}', 'Difference']:
                 if col in result_df.columns: result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            result_df.dropna(subset=[f'Actual {target_name}', f'Predicted {target_name}'], inplace=True)
        else:
            result_df['Difference'] = np.nan
            
        predictions_output[target_name] = result_df
    return predictions_output

# --- Main Application Flow ---
data = download_data(ticker)

if not data.empty:
    # Clear previous logs for this run
    training_messages_log.clear()
    training_messages_log.append(f"Data loaded for {ticker}: {len(data)} rows.")

    df_features_full = create_features(data.copy(), lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window)
    training_messages_log.append(f"Features created. Rows after NaN drop: {len(df_features_full)}")

    df_train_period = df_features_full[(df_features_full['Date'] >= pd.to_datetime(start_bt)) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()

    if df_train_period.empty:
        st.warning("No data in selected training range after feature creation. Adjust dates or check ticker.")
        training_messages_log.append("❌ No data in training range.")
        log_placeholder.text_area("Log:", "".join(f"{msg}\n" for msg in training_messages_log), height=200, disabled=True)
        st.stop()
    
    training_messages_log.append(f"Training data period: {len(df_train_period)} rows from {start_bt} to {end_bt}.")
    
    # Train main models
    st.markdown("---")
    st.subheader(f"📈 Main Model Training: {model_choice}")
    trained_models_main = train_models_pipeline(df_train_period.copy(), model_choice, perform_tuning)
    
    if not trained_models_main or not trained_models_main.get('Close', {}).get('model'):
        st.error("Main Close price model training failed. Check logs in sidebar.")
        log_placeholder.text_area("Log:", "".join(f"{msg}\n" for msg in training_messages_log), height=200, disabled=True)
        st.stop()

    # Display training logs
    log_placeholder.text_area("Log:", "".join(f"{msg}\n" for msg in training_messages_log), height=300, disabled=True)


    # --- Output 1: Predicted price for t2 + 1 ---
    last_day_features_df = df_train_period.tail(1).copy()
    if not last_day_features_df.empty:
        last_train_date = pd.to_datetime(end_bt)
        next_trading_day = last_train_date + timedelta(days=1)
        while next_trading_day.weekday() >= 5: next_trading_day += timedelta(days=1)

        st.markdown("---")
        st.subheader(f"🔮 Predicted Values for {next_trading_day.strftime('%Y-%m-%d')} (Next Trading Day)")
        
        next_day_predictions_list = []
        last_actual_close_full_data = data['Close'].iloc[-1] if not data.empty else None

        # Main Model Next Day Prediction
        next_day_preds_main_dict = generate_predictions_pipeline(last_day_features_df.copy(), trained_models_main)
        
        pred_close_main = next_day_preds_main_dict.get('Close', {}).get('Predicted Close', pd.Series()).iloc[-1] if not next_day_preds_main_dict.get('Close', {}).empty else None
        pred_open_main = next_day_preds_main_dict.get('Open', {}).get('Predicted Open', pd.Series()).iloc[-1] if not next_day_preds_main_dict.get('Open', {}).empty else None
        pred_vol_main = next_day_preds_main_dict.get('Volatility', {}).get('Predicted Volatility', pd.Series()).iloc[-1] if not next_day_preds_main_dict.get('Volatility', {}).empty else None

        next_day_predictions_list.append({'Model': model_choice, 'Predicted Close': pred_close_main, 'Predicted Open': pred_open_main, 'Predicted Volatility': pred_vol_main})
        if pred_close_main is not None:
            # For the next trading day prediction, training_end_date_used is naturally end_bt
            save_prediction(ticker, next_trading_day, pred_close_main, np.nan, model_choice, datetime.now(), end_bt)

        # Comparison Models Next Day Prediction
        for comp_model_name in compare_models:
            if comp_model_name == model_choice: continue # Skip if same as main
            # training_messages_log.append(f"Processing next day for comparison model: {comp_model_name}") # Logged in sidebar
            temp_train_log = [] # Local log for this model
            trained_comp_model_dict = train_models_pipeline(df_train_period.copy(), comp_model_name, perform_tuning) # Retrain for this specific model type
            
            if trained_comp_model_dict and trained_comp_model_dict.get('Close', {}).get('model'):
                next_day_preds_comp_dict = generate_predictions_pipeline(last_day_features_df.copy(), trained_comp_model_dict)
                pred_close_comp = next_day_preds_comp_dict.get('Close', {}).get('Predicted Close', pd.Series()).iloc[-1] if not next_day_preds_comp_dict.get('Close', {}).empty else None
                pred_open_comp = next_day_preds_comp_dict.get('Open', {}).get('Predicted Open', pd.Series()).iloc[-1] if not next_day_preds_comp_dict.get('Open', {}).empty else None
                pred_vol_comp = next_day_preds_comp_dict.get('Volatility', {}).get('Predicted Volatility', pd.Series()).iloc[-1] if not next_day_preds_comp_dict.get('Volatility', {}).empty else None
                next_day_predictions_list.append({'Model': comp_model_name, 'Predicted Close': pred_close_comp, 'Predicted Open': pred_open_comp, 'Predicted Volatility': pred_vol_comp})
                if pred_close_comp is not None:
                    # For the next trading day prediction, training_end_date_used is naturally end_bt
                    save_prediction(ticker, next_trading_day, pred_close_comp, np.nan, comp_model_name, datetime.now(), end_bt)
        
        df_next_day_preds = pd.DataFrame(next_day_predictions_list)
        if not df_next_day_preds.empty:
            df_next_day_preds['Date'] = next_trading_day
            df_next_day_preds = df_next_day_preds[['Date', 'Model', 'Predicted Close', 'Predicted Open', 'Predicted Volatility']].sort_values('Model').reset_index(drop=True)
            st.dataframe(df_next_day_preds.style.format({"Predicted Close": "{:.2f}", "Predicted Open": "{:.2f}", "Predicted Volatility": "{:.4f}"}))
            if pred_close_main is not None and last_actual_close_full_data is not None:
                 st.markdown(f"**Difference (Last Actual Close - Predicted Next Day Close) for Main Model ({model_choice}):** {(last_actual_close_full_data - pred_close_main):.2f}")
        else:
            st.warning("Could not generate next day predictions.")

    # --- Output 5: Model Comparison on Recent Data ---
    st.markdown("---")
    st.subheader("📊 Selected Models Comparison on Recent Data")
    compare_days_actual = min(train_days_comparison, len(df_features_full))
    df_compare_data = df_features_full.tail(compare_days_actual).copy()
    
    comparison_results_list = []
    best_comp_model_name, best_comp_rmse, best_comp_pct_rmse = "N/A", float('inf'), float('inf')

    if df_compare_data.empty:
        st.warning("Not enough data for model comparison chart.")
    else:
        avg_actual_compare = df_compare_data['Close'].mean()
        fig_compare_chart = go.Figure()
        fig_compare_chart.add_trace(go.Scatter(x=df_compare_data['Date'], y=df_compare_data['Close'], mode='lines', name='Actual Close', line=dict(color='black', width=2)))
        
        colors = px.colors.qualitative.Plotly
        
        # Add main model to comparison if not already there implicitly
        models_for_chart = compare_models[:] # Make a copy
        if model_choice not in models_for_chart:
            models_for_chart.insert(0, model_choice) # Add main model to the beginning for comparison

        for i, model_name_iter in enumerate(models_for_chart):
            # Use existing trained main model if it's the current iteration
            if model_name_iter == model_choice and 'Close' in trained_models_main and trained_models_main['Close']['model']:
                model_info_iter = {'Close': trained_models_main['Close']} # Use already trained main model
            else: # Train other comparison models
                # training_messages_log.append(f"Training {model_name_iter} for comparison chart...") # Logged in sidebar
                model_info_iter = train_models_pipeline(df_compare_data.copy(), model_name_iter, perform_tuning) # Train on comparison period

            if model_info_iter and model_info_iter.get('Close', {}).get('model'):
                preds_dict_iter = generate_predictions_pipeline(df_compare_data.copy(), model_info_iter)
                if 'Close' in preds_dict_iter and not preds_dict_iter['Close'].empty:
                    pred_df_iter = preds_dict_iter['Close']
                    if not pred_df_iter.empty and 'Actual Close' in pred_df_iter and 'Predicted Close' in pred_df_iter:
                        pred_df_iter.dropna(subset=['Actual Close', 'Predicted Close'], inplace=True)
                        if not pred_df_iter.empty:
                            y_actual_iter = pred_df_iter['Actual Close']
                            y_pred_iter = pred_df_iter['Predicted Close']
                            mae_iter = mean_absolute_error(y_actual_iter, y_pred_iter)
                            rmse_iter = np.sqrt(mean_squared_error(y_actual_iter, y_pred_iter))
                            pct_mae_iter = (mae_iter / avg_actual_compare) * 100 if avg_actual_compare > 0 else np.nan
                            pct_rmse_iter = (rmse_iter / avg_actual_compare) * 100 if avg_actual_compare > 0 else np.nan
                            
                            comparison_results_list.append({'Model': model_name_iter, 'MAE': mae_iter, 'RMSE': rmse_iter, '%-MAE': pct_mae_iter, '%-RMSE': pct_rmse_iter})
                            if rmse_iter < best_comp_rmse: best_comp_rmse, best_comp_model_name = rmse_iter, model_name_iter
                            if pct_rmse_iter < best_comp_pct_rmse: best_comp_pct_rmse = pct_rmse_iter
                                
                            fig_compare_chart.add_trace(go.Scatter(x=pred_df_iter['Date'], y=y_pred_iter, mode='lines', name=f"{model_name_iter} Pred.", line=dict(color=colors[i % len(colors)], dash='dot')))
        
        fig_compare_chart.update_layout(title=f"Model Comparison: Actual vs Predicted Close ({compare_days_actual} days)", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
        st.plotly_chart(fig_compare_chart, use_container_width=True)
        
        if comparison_results_list:
            df_comparison_tbl = pd.DataFrame(comparison_results_list).sort_values(['RMSE', '%-RMSE']).dropna(subset=['RMSE'])
            st.dataframe(df_comparison_tbl.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "%-MAE": "{:.2f}%", "%-RMSE": "{:.2f}%"}))
            if best_comp_model_name != "N/A":
                 st.markdown(f"🏆 **Best performing in comparison (lowest RMSE): {best_comp_model_name}** (RMSE: {best_comp_rmse:.4f}, %-RMSE: {best_comp_pct_rmse:.2f}%)")


    # --- Output 2: Training period actual vs predicted (Main Model) ---
    st.markdown("---")
    st.subheader(f"🎯 Training Period Performance: {model_choice} (Close Price)")
    train_preds_dict = generate_predictions_pipeline(df_train_period.copy(), {'Close': trained_models_main.get('Close')}) # Only Close for this chart

    if 'Close' in train_preds_dict and not train_preds_dict['Close'].empty:
        train_pred_df_main = train_preds_dict['Close']
        if not train_pred_df_main.empty and 'Actual Close' in train_pred_df_main:
            actual_train_main = train_pred_df_main['Actual Close']
            predicted_train_main = train_pred_df_main['Predicted Close']
            mae_train_main = mean_absolute_error(actual_train_main, predicted_train_main)
            rmse_train_main = np.sqrt(mean_squared_error(actual_train_main, predicted_train_main))
            avg_actual_train_main = actual_train_main.mean()
            pct_mae_train_main = (mae_train_main / avg_actual_train_main) * 100 if avg_actual_train_main > 0 else np.nan
            pct_rmse_train_main = (rmse_train_main / avg_actual_train_main) * 100 if avg_actual_train_main > 0 else np.nan
            st.markdown(f"**Metrics on Training Data ({start_bt.strftime('%Y-%m-%d')} to {end_bt.strftime('%Y-%m-%d')}):**")
            st.markdown(f"MAE: {mae_train_main:.4f} ({pct_mae_train_main:.2f}%) | RMSE: {rmse_train_main:.4f} ({pct_rmse_train_main:.2f}%)")

            fig_train_perf = go.Figure()
            fig_train_perf.add_trace(go.Scatter(x=train_pred_df_main['Date'], y=actual_train_main, mode='lines', name='Actual Close', line=dict(color='royalblue')))
            fig_train_perf.add_trace(go.Scatter(x=train_pred_df_main['Date'], y=predicted_train_main, mode='lines', name='Predicted Close', line=dict(color='orangered', dash='dash')))
            fig_train_perf.update_layout(title=f"Training: Actual vs Predicted Close ({model_choice})", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_train_perf, use_container_width=True)
        else: st.info("Not enough data in training predictions to plot/calculate metrics.")
    else: st.warning("Could not generate training period predictions for the main Close model.")

    # --- Output 3: Backtesting on last 30 days of training (Main Model) ---
    st.markdown("---")
    st.subheader(f"📉 Backtesting (Last 30 Trading Days of Training Period): {model_choice} (Close Price)")
    backtest_start_date_30d = pd.to_datetime(end_bt) - timedelta(days=45) # Fetch a bit more to ensure 30 trading days
    df_backtest_data_30d = df_features_full[(df_features_full['Date'] > backtest_start_date_30d) & (df_features_full['Date'] <= pd.to_datetime(end_bt))].copy()
    df_backtest_data_30d = df_backtest_data_30d.tail(30) # Get last 30 available days

    if not df_backtest_data_30d.empty:
        bt_preds_dict = generate_predictions_pipeline(df_backtest_data_30d.copy(), {'Close': trained_models_main.get('Close')})
        if 'Close' in bt_preds_dict and not bt_preds_dict['Close'].empty:
            bt_pred_df_main = bt_preds_dict['Close']
            if not bt_pred_df_main.empty and 'Actual Close' in bt_pred_df_main:
                actual_bt_main = bt_pred_df_main['Actual Close']
                predicted_bt_main = bt_pred_df_main['Predicted Close']
                mae_bt_main = mean_absolute_error(actual_bt_main, predicted_bt_main)
                rmse_bt_main = np.sqrt(mean_squared_error(actual_bt_main, predicted_bt_main))
                avg_actual_bt_main = actual_bt_main.mean()
                pct_mae_bt_main = (mae_bt_main / avg_actual_bt_main) * 100 if avg_actual_bt_main > 0 else np.nan
                pct_rmse_bt_main = (rmse_bt_main / avg_actual_bt_main) * 100 if avg_actual_bt_main > 0 else np.nan
                st.markdown(f"**Backtest Metrics (approx last 30 days of training):**")
                st.markdown(f"MAE: {mae_bt_main:.4f} ({pct_mae_bt_main:.2f}%) | RMSE: {rmse_bt_main:.4f} ({pct_rmse_bt_main:.2f}%)")
                st.dataframe(bt_pred_df_main[['Date', 'Actual Close', 'Predicted Close', 'Difference']].style.format({"Actual Close": "{:.2f}", "Predicted Close": "{:.2f}", "Difference": "{:.2f}"}))
                
                # training_messages_log.append(f"Saving backtest predictions for {ticker} (last 30 days)...") # Logged in sidebar
                for _, row in bt_pred_df_main.iterrows():
                    # For historical backtest predictions, training_end_date_used should be the day before the prediction_for_date
                    training_end_date_for_log = get_previous_trading_day(row['Date'])
                    save_prediction(ticker, row['Date'], row['Predicted Close'], row['Actual Close'], model_choice, datetime.now(), training_end_date_for_log)
            else: st.info("Not enough data in backtest predictions to display.")
        else: st.warning("Could not generate backtest predictions for the main Close model.")
    else: st.info("Not enough data for the 30-day backtest period.")


    # --- Output 4: Future Predictions (Iterative, Main Model Close Price) ---
    st.markdown("---")
    st.subheader(f"🚀 Future {n_future} Days Predicted Close Prices ({model_choice})")
    st.info("""
    **Iterative Forecasting Explained:** Monarch predicts future prices one day at a time. 
    The prediction for Day 1 is made. Then, this Day 1 prediction is used as an input (as if it were actual data) to help predict Day 2, and so on. 
    This process repeats for the number of future days you select. 
    *Remember: These are model-based projections and not financial advice.*
    """)

    future_predictions_output_list = []
    main_close_model_info = trained_models_main.get('Close')

    if main_close_model_info and main_close_model_info['model']:
        model_fut = main_close_model_info['model']
        scaler_fut = main_close_model_info['scaler']
        feature_cols_fut_train = main_close_model_info['features']
        
        # Need enough historical data to calculate features for the first future day
        max_hist_window = max(lag_features_list + ma_windows_list + std_windows_list + [rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window])
        
        df_hist_context_fut = data[data['Date'] <= pd.to_datetime(end_bt)].tail(max_hist_window + 5).copy() # +5 for buffer
        
        current_iter_date = pd.to_datetime(end_bt)

        for i in range(n_future):
            next_pred_date = current_iter_date + timedelta(days=1)
            while next_pred_date.weekday() >= 5: next_pred_date += timedelta(days=1)

            # Create features for the data up to current_iter_date
            df_features_for_pred = create_features(df_hist_context_fut.copy(), lag_features_list, ma_windows_list, std_windows_list, rsi_window, macd_short_window, macd_long_window, macd_signal_window, bb_window, bb_std_dev, atr_window, stoch_window, stoch_smooth_window)
            
            if df_features_for_pred.empty:
                # training_messages_log.append(f"⚠️ Empty features at future step {i+1}. Stopping.") # Logged in sidebar
                break
            
            last_feature_row = df_features_for_pred.tail(1)
            
            X_fut_predict = last_feature_row.reindex(columns=feature_cols_fut_train, fill_value=0)
            if X_fut_predict.empty or X_fut_predict.shape[1] != len(feature_cols_fut_train):
                # training_messages_log.append(f"⚠️ Feature mismatch at future step {i+1}. Stopping.") # Logged in sidebar
                break
            
            X_fut_scaled = scaler_fut.transform(X_fut_predict)
            try:
                predicted_price_fut = model_fut.predict(X_fut_scaled)[0]
            except Exception as e_fut:
                # training_messages_log.append(f"❌ Error at future step {i+1}: {e_fut}. Stopping.") # Logged in sidebar
                break

            future_predictions_output_list.append({'Date': next_pred_date, 'Predicted Close': predicted_price_fut})
            
            # Append predicted row to historical context for next iteration
            # Use last known OHLV for simplicity, Close is the new prediction
            new_row_simulated = pd.DataFrame([{
                'Date': next_pred_date,
                'Open': df_hist_context_fut['Open'].iloc[-1], # Simplification
                'High': predicted_price_fut, # Simplification: High as predicted close
                'Low': predicted_price_fut,  # Simplification: Low as predicted close
                'Close': predicted_price_fut,
                'Volume': df_hist_context_fut['Volume'].iloc[-1] # Simplification
            }])
            df_hist_context_fut = pd.concat([df_hist_context_fut, new_row_simulated], ignore_index=True).tail(max_hist_window + 5)
            current_iter_date = next_pred_date
            
        if future_predictions_output_list:
            df_future_preds_tbl = pd.DataFrame(future_predictions_output_list)
            st.dataframe(df_future_preds_tbl.style.format({"Predicted Close": "{:.2f}"}))
            fig_future_chart = px.line(df_future_preds_tbl, x='Date', y='Predicted Close', title=f"Future {n_future} Days Predicted Close ({model_choice})")
            fig_future_chart.update_traces(line=dict(color='mediumseagreen'))
            st.plotly_chart(fig_future_chart, use_container_width=True)
            # training_messages_log.append(f"Saving {len(future_predictions_output_list)} future predictions...") # Logged in sidebar
            for pred_item in future_predictions_output_list:
                # For future predictions, training_end_date_used should be the day before the prediction_for_date
                training_end_date_for_log = get_previous_trading_day(pred_item['Date'])
                save_prediction(ticker, pred_item['Date'], pred_item['Predicted Close'], np.nan, model_choice, datetime.now(), training_end_date_for_log)
        else:
            st.warning("Could not generate future predictions. Check logs.")
    else:
        st.warning("Main Close model not available for future predictions.")


    # --- Historical Prediction Log Display ---
    st.markdown("---")
    st.header("🕰️ Historical Prediction Log")
    st.markdown("""
    This log shows all predictions made by Monarch for this ticker. 
    `prediction_generation_date`: When the prediction was made.
    `prediction_for_date`: The date the prediction was *for*.
    `training_end_date_used`: The 'Training End Date (t2)' used for the model that made this prediction.
    This helps track how forecasts for a specific date changed as more data became available.
    """)
    past_preds_df = load_past_predictions(ticker)
    if not past_preds_df.empty:
        # Ensure all expected columns exist, adding them with NaT/NaN if not (for backward compatibility with old logs)
        expected_log_cols = ['prediction_generation_date', 'prediction_for_date', 'training_end_date_used', 'ticker', 'model_used', 'predicted_value', 'actual_close']
        for col in expected_log_cols:
            if col not in past_preds_df.columns:
                if 'date' in col: past_preds_df[col] = pd.NaT
                else: past_preds_df[col] = np.nan
        
        past_preds_df_display = past_preds_df.sort_values(by=['prediction_for_date', 'prediction_generation_date', 'training_end_date_used'], ascending=[False, False, False]).reset_index(drop=True)
        
        # Select and reorder columns for display
        display_cols = ['prediction_for_date', 'predicted_value', 'actual_close', 'model_used', 'prediction_generation_date', 'training_end_date_used']
        past_preds_df_display = past_preds_df_display[display_cols]

        st.dataframe(past_preds_df_display.style.format({
            "predicted_value": "{:.2f}", "actual_close": "{:.2f}",
            "prediction_generation_date": lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'N/A',
            "prediction_for_date": lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else 'N/A',
            "training_end_date_used": lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else 'N/A'
        }))

        st.subheader("Visualizing Past Predictions vs. Actuals")
        plot_df_past = past_preds_df_display.dropna(subset=['actual_close', 'predicted_value', 'prediction_for_date']).copy()
        if not plot_df_past.empty:
            fig_past_log = go.Figure()
            actual_data_log_plot = plot_df_past.groupby('prediction_for_date')['actual_close'].first().reset_index()
            fig_past_log.add_trace(go.Scatter(x=actual_data_log_plot['prediction_for_date'], y=actual_data_log_plot['actual_close'], mode='lines+markers', name='Actual Close', line=dict(color='navy', width=2)))

            # Plot predictions, perhaps only the latest one for each 'prediction_for_date' or a selection
            # For simplicity, let's plot all logged predictions. Could get crowded.
            # Consider allowing user to filter which generation dates to show.
            unique_gen_dates = plot_df_past['prediction_generation_date'].dt.date.unique()
            limited_gen_dates = sorted(unique_gen_dates, reverse=True)[:5] # Show last 5 generation days for clarity

            for gen_d in limited_gen_dates:
                gen_df_subset = plot_df_past[plot_df_past['prediction_generation_date'].dt.date == gen_d]
                if not gen_df_subset.empty:
                     # For each generation date, if multiple models predicted, show them
                    for model_log_name in gen_df_subset['model_used'].unique():
                        model_gen_subset = gen_df_subset[gen_df_subset['model_used'] == model_log_name]
                        fig_past_log.add_trace(go.Scatter(
                            x=model_gen_subset['prediction_for_date'], y=model_gen_subset['predicted_value'],
                            mode='lines+markers', name=f'Pred ({model_log_name}) Gen: {gen_d.strftime("%Y-%m-%d")}',
                            line=dict(dash='dot') # Differentiate predictions
                        ))
            
            fig_past_log.update_layout(title=f'Historical Log: Predicted vs. Actual Close for {ticker} (Recent Generations)', xaxis_title="Date of Stock Price", yaxis_title="Price", hovermode="x unified")
            st.plotly_chart(fig_past_log, use_container_width=True)
        else: st.info("No past predictions with known actuals to plot from log.")
    else: st.info(f"No past prediction logs found for {ticker}.")


    # --- Feature Importance (Main Close Model) ---
    st.markdown("---")
    st.subheader(f"🧠 Feature Importance / Coefficients ({model_choice} - Close Price Model)")
    main_close_model_info_fi = trained_models_main.get('Close')
    if main_close_model_info_fi and main_close_model_info_fi.get('model') and main_close_model_info_fi.get('features'):
        model_fi = main_close_model_info_fi['model']
        feature_names_fi = main_close_model_info_fi['features']
        
        fi_data = None
        if hasattr(model_fi, 'feature_importances_'):
            fi_data = pd.DataFrame({'Feature': feature_names_fi, 'Importance': model_fi.feature_importances_}).sort_values('Importance', ascending=False)
            chart_title_fi = f'Feature Importances for {model_choice} (Close Price)'
            bar_x_fi, bar_y_fi = 'Importance', 'Feature'
        elif hasattr(model_fi, 'coef_'):
            fi_data = pd.DataFrame({'Feature': feature_names_fi, 'Coefficient': model_fi.coef_}).sort_values('Coefficient', ascending=False)
            chart_title_fi = f'Coefficients for {model_choice} (Close Price)'
            bar_x_fi, bar_y_fi = 'Coefficient', 'Feature'
        
        if fi_data is not None:
            st.dataframe(fi_data.head(15)) # Show top 15
            fig_fi_chart = px.bar(fi_data.head(15), x=bar_x_fi, y=bar_y_fi, orientation='h', title=chart_title_fi)
            st.plotly_chart(fig_fi_chart, use_container_width=True)
        else: st.info(f"Feature importance/coefficients not available for {model_choice}.")
    else: st.info("Main Close model details not available for feature importance.")

else:
    st.warning(f"Could not load data for {ticker}. Please check the ticker symbol and your internet connection.")

st.markdown("---")
st.caption("Monarch Stock Price Predictor | Disclaimer: For educational and informational purposes only. Not financial advice")
