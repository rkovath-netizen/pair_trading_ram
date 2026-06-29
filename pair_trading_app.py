import os
import time

# ==============================================================================
# CRITICAL TIMEZONE CORRECTION HOOK (MUST BE RUN BEFORE ANY OTHER IMPORTS)
# ==============================================================================
# Forces the entire online Streamlit Cloud Linux container to run on Indian Standard Time
os.environ['TZ'] = 'Asia/Kolkata'
if hasattr(time, 'tzset'):
    time.tzset()

import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta, time as datetime_time
import holidays
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import subprocess

# --- INITIAL STRATEGY PARAMETERS ---
API_TOKEN = st.secrets["UPSTOX_API_TOKEN"]
SECTOR_CSV = "fno_with_sectors.csv"     
FORWARD_LOG_CSV = "forward_test_log.csv"  

# Extract Github Deployment Secrets Vault
GH_PAT = st.secrets["GITHUB_PAT"]
GH_USER = st.secrets["GITHUB_USERNAME"]
GH_REPO = st.secrets["GITHUB_REPO_NAME"]

LOOKBACK_WINDOW = 80  
P_VAL_THRESHOLD = 0.10  
Z_ENTRY_LIMIT = 1.5     
Z_STOP_LOSS = 2.5       
MAX_BAR_DURATION = 80     

st.set_page_config(page_title="StatArb Trading Desk", layout="wide")
st_autorefresh(interval=10000, key="deskrefresh")

st.title("📊 Live Statistical Arbitrage Processing Desk")
st.markdown("---")

headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {API_TOKEN}'
}

# --- THE AUTO-COMMIT GITHUB REPOSITORY PUSHER ---
def git_push_to_github():
    """Executes a structural terminal push back to the main GitHub repository node using IST tracking."""
    try:
        # Authenticate git context via the token link URL string
        remote_url = f"https://{GH_USER}:{GH_PAT}@github.com/{GH_USER}/{GH_REPO}.git"
        
        # Sequentially execute standard shell pipeline terminal actions
        subprocess.run(["git", "config", "user.name", "Streamlit Live Bot"], check=True)
        subprocess.run(["git", "config", "user.email", "bot@streamlit.io"], check=True)
        subprocess.run(["git", "add", FORWARD_LOG_CSV], check=True)
        
        # Commit message will now correctly stamp in your local execution clock time
        commit_msg = f"Automated Forward Log Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", remote_url, "main"], check=True)
    except Exception as e:
        # Prevent any git sync disruptions from hanging your active live scanning cycle
        pass

def append_to_forward_log(trade_data):
    file_exists = os.path.exists(FORWARD_LOG_CSV)
    df_new = pd.DataFrame([trade_data])
    df_new.to_csv(FORWARD_LOG_CSV, mode='a', header=not file_exists, index=False)
    
    # Trigger background auto-push to your GitHub folder instantly
    git_push_to_github()

# --- TRACKING STATES MANAGED PER SPECIFIC BROWSER SESSION ---
if 'active_trades' not in st.session_state:
    st.session_state.active_trades = {}
if 'closed_trades_today' not in st.session_state:
    st.session_state.closed_trades_today = []

indian_holidays = holidays.India(years=[datetime.today().year])

# --- MATHEMATICAL UTILITIES ---
def get_next_trading_day(from_date):
    next_day = from_date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.strftime("%Y-%m-%d") in indian_holidays:
        next_day += timedelta(days=1)
    return datetime.combine(next_day, datetime_time(9, 15, 0))

def check_market_status():
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5:
        next_open = get_next_trading_day(now)
        return False, f"Weekend ({now.strftime('%A')})", next_open
    if today_str in indian_holidays:
        next_open = get_next_trading_day(now)
        return False, f"Holiday: {indian_holidays.get(today_str)}", next_open
    market_open_today = datetime.combine(now.date(), datetime_time(9, 15, 0))
    if now < market_open_today:
        return False, "Waiting for 9:15 AM Open", market_open_today
    market_close_today = datetime.combine(now.date(), datetime_time(15, 30, 0))
    if now > market_close_today:
        next_open = get_next_trading_day(now)
        return False, "Market Closed", next_open
    return True, "MARKET ACTIVE", None

market_is_open, log_message, next_open_dt = check_market_status()

# Render System Status Cards
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("System Operational Node", "ACTIVE" if market_is_open else "SUSPENDED")
# This clock will now tick beautifully in true Indian Standard Time online
col_s2.metric("Pipeline Clock", datetime.now().strftime("%H:%M:%S"))
col_s3.metric("Operational Context", log_message if not market_is_open else "15m Intraday Continuous Sync")

st.markdown("---")

# --- BACKGROUND ARBITRAGE SCAN PATHWAY ---
if market_is_open:
    sector_df = pd.read_csv(SECTOR_CSV)
    sector_df.columns = sector_df.columns.str.strip()
    sector_df['Symbol'] = sector_df['Symbol'].str.strip().str.upper()
    sector_df['Sector'] = sector_df['Sector'].str.strip()
    sector_groups = sector_df.groupby('Sector')['Symbol'].apply(list).to_dict()

    def build_upstox_v3_key_map():
        url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
        try:
            df_master = pd.read_csv(url, compression='gzip')
            df_master.columns = df_master.columns.str.strip().str.lower()
            sym_col = [c for c in df_master.columns if 'symbol' in c or 'ticker' in c][0]
            key_col = [c for c in df_master.columns if 'key' in c or 'instrument' in c][0]
            return {str(row[sym_col]).strip().upper(): str(row[key_col]).strip() for _, row in df_master.iterrows() if "NSE_EQ" in str(row[key_col])}
        except Exception:
            return {sym: f"NSE_EQ|{sym}" for sym in sector_df['Symbol'].unique()}

    upstox_key_map = build_upstox_v3_key_map()

    def fetch_complete_series_15min(symbol):
        api_key = upstox_key_map.get(symbol)
        if not api_key: return pd.Series(dtype=float)
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=15)
        hist_url = f"https://api.upstox.com/v3/historical-candle/{api_key}/minutes/15/{end_dt.strftime('%Y-%m-%d')}/{start_dt.strftime('%Y-%m-%d')}"
        res_hist = requests.get(hist_url, headers=headers)
        live_url = f"https://api.upstox.com/v3/historical-candle/intraday/{api_key}/minutes/15"
        res_live = requests.get(live_url, headers=headers)
        candles = []
        if res_hist.status_code == 200: candles.extend(res_hist.json().get('data', {}).get('candles', []))
        if res_live.status_code == 200: candles.extend(res_live.json().get('data', {}).get('candles', []))
        if not candles: return pd.Series(dtype=float)
        df = pd.DataFrame(candles, columns=['Time', 'O', 'H', 'L', 'C', 'V', 'OI'])
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(ascending=True, inplace=True)
        return df['C']

    price_dict = {}
    for sym in sector_df['Symbol'].unique():
        series = fetch_complete_series_15min(sym)
        if not series.empty: price_dict[sym] = series
    df_prices = pd.DataFrame(price_dict).dropna()

    if not df_prices.empty:
        for p_key in list(st.session_state.active_trades.keys()):
            st.session_state.active_trades[p_key]['bars_in_trade'] += 1

        for sector, symbols in sector_groups.items():
            valid_syms = [s for s in symbols if s in df_prices.columns]
            for i in range(len(valid_syms)):
                for j in range(i + 1, len(valid_syms)):
                    stock_A, stock_B = valid_syms[i], valid_syms[j]
                    pair_key = f"{stock_A}_{stock_B}"
                    
                    Y, X = df_prices[stock_A], df_prices[stock_B]
                    X_const = sm.add_constant(X)
                    if adfuller(sm.OLS(Y, X_const).fit().resid)[1] > P_VAL_THRESHOLD: continue
                        
                    model = sm.OLS(Y, X_const).fit()
                    beta = model.params[stock_B]
                    spread = Y - (beta * X)
                    
                    mean_spread = spread.rolling(LOOKBACK_WINDOW).mean().iloc[-1]
                    std_spread = spread.rolling(LOOKBACK_WINDOW).std().iloc[-1]
                    current_z = (spread.iloc[-1] - mean_spread) / std_spread
                    
                    pA_now, pB_now = Y.iloc[-1], X.iloc[-1]
                    lot_A = sector_df.loc[sector_df['Symbol'] == stock_A, 'lot size'].values[0]
                    lot_B = sector_df.loc[sector_df['Symbol'] == stock_B, 'lot size'].values[0]
                    
                    if pair_key in st.session_state.active_trades:
                        trade = st.session_state.active_trades[pair_key]
                        is_exit = False
                        exit_reason = ""
                        
                        pnl_current_A = (trade['qty_A'] * (trade['entry_pA'] - pA_now)) if trade['direction'] == "SHORT" else (trade['qty_A'] * (pA_now - trade['entry_pA']))
                        pnl_current_B = (trade['qty_B'] * (pB_now - trade['entry_pB'])) if trade['direction'] == "SHORT" else (trade['qty_B'] * (trade['entry_pB'] - pB_now))
                        running_pnl = pnl_current_A + pnl_current_B
                        
                        if not trade['has_scaled_partial']:
                            reached_partial = (trade['direction'] == "SHORT" and current_z <= 0.5) or (trade['direction'] == "LONG" and current_z >= -0.5)
                            if reached_partial:
                                trade['has_scaled_partial'] = True
                                trade['locked_partial_pnl'] = running_pnl * 0.70
                                trade['qty_A'] = round(trade['qty_A'] * 0.30)
                                trade['qty_B'] = round(trade['qty_B'] * 0.30)
                                
                                append_to_forward_log({
                                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'Pair ID': pair_key, 'Event': 'PARTIAL_PROFIT_TAKEN',
                                    'Z-Score': round(current_z, 2), 'Current PnL': round(running_pnl, 2)
                                })
                        
                        if (trade['direction'] == "SHORT" and current_z <= 0.0) or (trade['direction'] == "LONG" and current_z >= 0.0):
                            is_exit, exit_reason = True, "Target Hit (Z=0)"
                        elif (trade['direction'] == "SHORT" and current_z >= Z_STOP_LOSS) or (trade['direction'] == "LONG" and current_z <= -Z_STOP_LOSS):
                            is_exit, exit_reason = True, "Hard Stop Loss Triggered"
                        elif trade['bars_in_trade'] >= MAX_BAR_DURATION:
                            is_exit, exit_reason = True, "Max Bar Timeout Reached"
                            
                        if is_exit:
                            final_realized_pnl = trade['locked_partial_pnl'] + running_pnl if trade['has_scaled_partial'] else running_pnl
                            
                            append_to_forward_log({
                                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'Pair ID': pair_key, 'Event': f'EXIT_{exit_reason.replace(" ", "_").upper()}',
                                'Z-Score': round(current_z, 2), 'Current PnL': round(final_realized_pnl, 2)
                            })
                            
                            st.session_state.closed_trades_today.append({
                                'pair': pair_key, 'direction': trade['direction'], 'pnl': final_realized_pnl,
                                'bars': trade['bars_in_trade'], 'capital': trade['cap_A'] + trade['cap_B']
                            })
                            del st.session_state.active_trades[pair_key]
                    else:
                        if abs(current_z) >= Z_ENTRY_LIMIT and abs(current_z) < Z_STOP_LOSS:
                            direction = "SHORT" if current_z >= Z_ENTRY_LIMIT else "LONG"
                            raw_ratio = beta * (pB_now / pA_now)
                            qty_B = max(1, round((lot_A * raw_ratio) / lot_B)) * lot_B
                            cap_A, cap_B = lot_A * pA_now, qty_B * pB_now
                            
                            st.session_state.active_trades[pair_key] = {
                                'direction': direction, 'entry_pA': pA_now, 'entry_pB': pB_now,
                                'qty_A': lot_A, 'qty_B': qty_B, 'cap_A': cap_A, 'cap_B': cap_B,
                                'bars_in_trade': 0, 'current_z': round(current_z, 2),
                                'has_scaled_partial': False, 'locked_partial_pnl': 0.0
                            }
                            
                            append_to_forward_log({
                                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'Pair ID': pair_key, 'Event': f'ENTRY_{direction}',
                                'Z-Score': round(current_z, 2), 'Current PnL': 0.0
                            })

# --- RENDER DASHBOARD INTERFACES ---
st.subheader("⚡ Active Trades Execution Desk")
if not st.session_state.active_trades:
    st.info("All scanned assets currently stable within normal statistical parameters.")
else:
    active_rows = []
    for k, v in st.session_state.active_trades.items():
        active_rows.append({
            "Pair Identity": k, "Direction": v['direction'],
            "Entry Price A": v['entry_pA'], "Entry Price B": v['entry_pB'],
            "Duration": f"{v['bars_in_trade']} / 80 Bars",
            "Scaled Profit": "Yes (70% Locked)" if v['has_scaled_partial'] else "No",
            "Live Z": v['current_z']
        })
    st.dataframe(pd.DataFrame(active_rows), use_container_width=True)

st.markdown("---")
st.subheader("📈 Intraday Account Summary Statistics")
total_live_trades = len(st.session_state.closed_trades_today) + len(st.session_state.active_trades)

if total_live_trades == 0:
    st.warning("Awaiting market entry triggers to log historical metrics summaries.")
else:
    all_executed = st.session_state.closed_trades_today + [{'capital': v['cap_A'] + v['cap_B'], 'bars': v['bars_in_trade']} for v in st.session_state.active_trades.values()]
    total_inv = sum([t['capital'] for t in all_executed])
    realized_pnl = sum([t['pnl'] for t in st.session_state.closed_trades_today])
    wins = len([t for t in st.session_state.closed_trades_today if t['pnl'] > 0])
    live_winrate = (wins / len(st.session_state.closed_trades_today) * 100) if st.session_state.closed_trades_today else 0.0
    avg_live_bars = np.mean([t['bars'] for t in all_executed])
    pnl_percentage = (realized_pnl / total_inv * 100) if total_inv > 0 else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Executed Trades", total_live_trades)
    m2.metric("Capital Exposure", f"₹{total_inv:,.2f}")
    m3.metric("Realized PnL Today", f"₹{realized_pnl:,.2f}", delta=f"{pnl_percentage:+.2f}%")
    m4.metric("Session Win Rate", f"{live_winrate:.1f}%")
    m5.metric("Avg Duration", f"{avg_live_bars:.1f} Bars")

st.markdown("---")
st.subheader("📂 Permanent GitHub Repository Log View (`forward_test_log.csv`)")
if os.path.exists(FORWARD_LOG_CSV):
    df_log = pd.read_csv(FORWARD_LOG_CSV)
    st.dataframe(df_log.tail(20), use_container_width=True)
else:
    st.info("Awaiting the first operational live trade signal to initialize the GitHub log.")
