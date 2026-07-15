import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from zoneinfo import ZoneInfo

# ==========================================
# 0. セッション状態（データ保持）の初期化
# ==========================================
if 'target_ticker' not in st.session_state:
    st.session_state.target_ticker = 'XOM'
if 'screening_run' not in st.session_state:
    st.session_state.screening_run = False
if 'hit_tickers' not in st.session_state:
    st.session_state.hit_tickers = []
if 'last_scan_date' not in st.session_state:
    st.session_state.last_scan_date = ""

# ==========================================
# 1. ページの基本設定
# ==========================================
st.set_page_config(page_title="トレード検証＆探索アプリ", layout="wide")
st.title('📈 トレード戦略 検証＆探索アプリ (MACD & RSI)')

# タブの作成
tab1, tab2 = st.tabs(["📊 過去の勝率検証 (単一銘柄)", "🔎 今日のチャンス探索 (大量スクリーニング)"])

# ==========================================
# 自動リスト取得用の関数（キャッシュ化して高速化）
# ==========================================
@st.cache_data(ttl=86400)
def get_sp500():
    try:
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        df = pd.read_csv(url)
        return [str(t).replace('.', '-') for t in df['Symbol'].tolist()]
    except Exception:
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'JNJ']

@st.cache_data(ttl=86400)
def get_ndx100():
    return ['AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN', 'AMAT', 'AMD', 
            'AMGN', 'AMZN', 'ANSS', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CCEP', 
            'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 
            'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG', 
            'FAST', 'FTNT', 'GEHC', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 
            'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 
            'MELI', 'META', 'MNST', 'MSTR', 'MU', 'NFLX', 'NXPI', 'ODFL', 'ON', 'ORLY', 
            'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 
            'SBUX', 'SIRI', 'SNPS', 'SPLK', 'TEAM', 'TMUS', 'TSLA', 'TTD', 'TXN', 'VRSK', 
            'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZS']

@st.cache_data(ttl=86400)
def get_dow30():
    return ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 
            'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
            'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']

# ==========================================
# ★超高速一括スクリーニング関数（キャッシュ機能付き）
# ==========================================
@st.cache_data(ttl=86400)
def run_fast_screening(ticker_list, scan_rsi_threshold, scan_rsi_period, date_str):
    hit_tickers = []
    
    raw_data = yf.download(ticker_list, period='3mo', group_by='ticker', threads=True)
    
    for t in ticker_list:
        try:
            if len(ticker_list) == 1:
                scan_data = raw_data
            else:
                if t not in raw_data.columns.levels[0]:
                    continue
                scan_data = raw_data[t].dropna()
                
            if scan_data.empty or len(scan_data) < 30:
                continue
            
            exp1 = scan_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = scan_data['Close'].ewm(span=26, adjust=False).mean()
            scan_data['MACD'] = exp1 - exp2
            scan_data['Signal'] = scan_data['MACD'].ewm(span=9, adjust=False).mean()
            
            delta = scan_data['Close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/scan_rsi_period, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/scan_rsi_period, adjust=False).mean()
            rs = gain / loss
            scan_data['RSI'] = 100 - (100 / (1 + rs))
            
            latest = scan_data.iloc[-1]
            prev = scan_data.iloc[-2]
            
            # 共通条件：RSIが基準値以下であること
            cond_rsi = latest['RSI'] <= scan_rsi_threshold
            
            # 「クロス直前(しそう)」の条件
            cond_under = latest['MACD'] < latest['Signal']
            diff_latest = latest['Signal'] - latest['MACD']
            diff_prev = prev['Signal'] - prev['MACD']
            cond_closing = diff_latest < diff_prev
            cond_macd_up = latest['MACD'] > prev['MACD']
            is_approaching = cond_under and cond_closing and cond_macd_up
            
            # 「クロス直後(した)」の条件（昨日は下だったが、今日は上抜けた）
            is_crossed = (latest['MACD'] > latest['Signal']) and (prev['MACD'] <= prev['Signal'])
            
            # どちらかの条件を満たしていればリストに追加
            if cond_rsi and (is_approaching or is_crossed):
                status = "🟢 クロス直後!" if is_crossed else "🟡 クロス直前"
                hit_tickers.append({
                    '銘柄コード': t,
                    '状態': status,
                    '最新株価($)': float(latest['Close']),
                    '最新RSI': float(latest['RSI']),
                    'MACD': float(latest['MACD']),
                    'シグナル線': float(latest['Signal'])
                })
        except Exception:
            pass
            
    return hit_tickers

# ==========================================
# 2. サイドバー（検証タブ用の設定）
# ==========================================
st.sidebar.header('⚙️ 検証タブ用 基本設定')
input_val = st.sidebar.text_input('検証する銘柄コード (例: XOM, AAPL)', value=st.session_state.target_ticker)
st.session_state.target_ticker = input_val
ticker = st.session_state.target_ticker

days_later = st.sidebar.slider('何日後のリターンを計算する？', min_value=1, max_value=30, value=5)
period = st.sidebar.selectbox('データ取得期間', ['6mo', '1y', '2y', '5y', 'max'], index=1)

st.sidebar.header('🧠 戦略の選択')
strategy = st.sidebar.selectbox('検証するシグナル（買い条件）', [
    '1. MACD ゴールデンクロス', 
    '2. RSI 売られすぎからの反発',
