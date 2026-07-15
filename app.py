import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from zoneinfo import ZoneInfo

# ==========================================
# 0. セッション状態（データ保持）の初期化
# ==========================================
# これにより、銘柄をタップして画面が再描画されてもスクリーニング結果が消えなくなります
if 'calc_ticker' not in st.session_state:
    st.session_state.calc_ticker = 'XOM'
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
@st.cache_data(ttl=86400) # 1日キャッシュ
def run_fast_screening(ticker_list, scan_rsi_threshold, scan_rsi_period, date_str):
    hit_tickers = []
    
    # 500銘柄の3ヶ月分データを1回のAPIコールで一括ダウンロード
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
            
            # MACD
            exp1 = scan_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = scan_data['Close'].ewm(span=26, adjust=False).mean()
            scan_data['MACD'] = exp1 - exp2
            scan_data['Signal'] = scan_data['MACD'].ewm(span=9, adjust=False).mean()
            
            # RSI
            delta = scan_data['Close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/scan_rsi_period, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/scan_rsi_period, adjust=False).mean()
            rs = gain / loss
            scan_data['RSI'] = 100 - (100 / (1 + rs))
            
            latest = scan_data.iloc[-1]
            prev = scan_data.iloc[-2]
            
            # 条件判定
            cond_rsi = latest['RSI'] <= scan_rsi_threshold
            cond_under = latest['MACD'] < latest['Signal']
            diff_latest = latest['Signal'] - latest['MACD']
            diff_prev = prev['Signal'] - prev['MACD']
            cond_closing = diff_latest < diff_prev
            cond_macd_up = latest['MACD'] > prev['MACD']
            
            if cond_rsi and cond_under and cond_closing and cond_macd_up:
                hit_tickers.append({
                    '銘柄コード': t,
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
# セッション状態(calc_ticker)と連動させることで、プログラム側から値を書き換え可能にします
ticker = st.sidebar.text_input('検証する銘柄コード (例: XOM, AAPL)', key='calc_ticker')
days_later = st.sidebar.slider('何日後のリターンを計算する？', min_value=1, max_value=30, value=5)
period = st.sidebar.selectbox('データ取得期間', ['6mo', '1y', '2y', '5y', 'max'], index=1)

st.sidebar.header('🧠 戦略の選択')
strategy = st.sidebar.selectbox('検証するシグナル（買い条件）', [
    '1. MACD ゴールデンクロス', 
    '2. RSI 売られすぎからの反発', 
    '3. MACDクロス + RSI売られすぎ (合わせ技)',
    '4. RSIが基準値以下 (売られすぎ状態でのエントリー)'
])

st.sidebar.markdown('---')
st.sidebar.markdown('**パラメータ微調整**')
rsi_period = st.sidebar.number_input('RSI 期間', value=14, step=1)
rsi_threshold = st.sidebar.number_input('RSI 基準値 (この数値以下を売られすぎと判断)', value=40, step=1)

# ==========================================
# 3. タブ1: 過去の勝率検証
# ==========================================
with tab1:
    st.markdown('サイドバーで銘柄と戦略を選んで、過去のデータに基づいた勝率とリターンを検証します。')
    
    if st.button('検証スタート！', key='backtest_btn'):
        with st.spinner(f'{ticker} のデータを取得・計算中...'):
            data = yf.Ticker(ticker).history(period=period)
            
            if data.empty:
                st.error("データの取得に失敗しました。銘柄コードを確認してください。")
            else:
                data.index = data.index.tz_localize(None)
                
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                
                delta = data['Close'].diff()
                gain = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
                loss = -delta.clip(upper=0).ewm(alpha=1/rsi_period, adjust=False).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                macd_cross = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
                rsi_rebound = (data['RSI'] > rsi_threshold) & (data['RSI'].shift(1) <= rsi_threshold)
                rsi_is_low = data['RSI'] <= rsi_threshold
                
                if '1. MACD' in strategy:
                    data['Buy_Signal'] = macd_cross
                elif '2. RSI' in strategy:
                    data['Buy_Signal'] = rsi_rebound
                elif '3. MACDクロス + RSI' in strategy:
                    data['Buy_Signal'] = macd_cross & rsi_is_low
                elif '4. RSIが基準値以下' in strategy:
                    data['Buy_Signal'] = rsi_is_low
                
                data['Price_Later'] = data['Close'].shift(-days_later)
                data['Return_(%)'] = ((data['Price_Later'] - data['Close']) / data['Close']) * 100
                
                signal_events = data[data['Buy_Signal']].copy()
                valid_returns = signal_events['Return_(%)'].dropna()
                
                st.subheader(f'📊 検証結果: {ticker} (シグナル発生から {days_later} 営業日後)')
                st.markdown(f"**選択した戦略:** {strategy}")
                
                if len(valid_returns) > 0:
                    win_rate = (valid_returns > 0).sum() / len(valid_returns) * 100
                    avg_return = valid_returns.mean()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("シグナル発生回数", f"{len(valid_returns)} 回")
                    col2.metric("勝率", f"{win_rate:.1f} %")
                    col3.metric("平均リターン", f"{avg_return:+.2f} %")
                    
                    st.markdown("---")
                    
                    st.markdown('**チャート推移とシグナル発生ポイント**')
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1, 1]})
                    
                    ax1.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.7)
                    signal_dates = signal_events.index
                    signal_prices = signal_events['Close']
                    ax1.scatter(signal_dates, signal_prices, marker='^', color='red', s=100, label='Buy Signal', zorder=5)
                    ax1.set_ylabel('Price')
                    ax1.legend()
                    ax1.grid(True)
                    
                    ax2.plot(data.index, data['MACD'], label='MACD', color='blue')
                    ax2.plot(data.index, data['Signal'], label='Signal', color='red', linestyle='--')
                    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
                    ax2.set_ylabel('MACD')
                    ax2.legend(loc='upper left')
                    ax2.grid(True)
                    
                    ax3.plot(data.index, data['RSI'], label=f'RSI ({rsi_period})', color='purple')
                    ax3.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
                    ax3.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
                    ax3.axhline(rsi_threshold, color='orange', linestyle=':', alpha=0.8, label=f'Your Threshold ({rsi_threshold})')
                    ax3.set_xlabel('Date')
                    ax3.set_ylabel('RSI')
                    ax3.set_ylim(0, 100)
                    ax3.legend(loc='upper left')
                    ax3.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    
                    st.markdown('**詳細データ一覧**')
                    display_df = pd.DataFrame({
                        '購入時の株価': signal_events['Close'].round(2),
                        f'{days_later}日後の株価': signal_events['Price_Later'].round(2),
                        'リターン(%)': signal_events['Return_(%)'].round(2)
                    })
                    display_df.index = display_df.index.strftime('%Y-%m-%d')
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("指定された期間内にシグナルが一度も発生していないか、検証データが不足しています。条件を緩めてみてください。")

# ==========================================
# 4. タブ2: 今日のチャンス探索（スクリーニング）
# ==========================================
with tab2:
    st.markdown('### 🔎 米国株 大量スクリーニング (MACDクロス直前 × RSI売られすぎ)')
    st.markdown('厳しい条件を設定し、膨大な銘柄の中から「まさに今が仕込み時」のお宝銘柄を抽出します。')
    
    # ターゲット選択
    target_group = st.radio(
        "スクリーニング対象のリストを選択してください",
        ("🇺🇸 S&P 500 (約500銘柄)", "🇺🇸 NASDAQ 100 (ハイテク代表銘柄)", "🇺🇸 ダウ30種 (主要30銘柄)", "✍️ 自分で入力する"),
        horizontal=True
    )
    
    ticker_list = []
    if target_group == "✍️ 自分で入力する":
        default_tickers = 'AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META'
        tickers_input = st.text_area('監視する銘柄コード (カンマ区切り)', default_tickers)
        ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
    elif target_group == "🇺🇸 S&P 500 (約500銘柄)":
        ticker_list = get_sp500()
        st.info(f"S&P 500 構成銘柄 ({len(ticker_list)}銘柄) をスキャンします。※初回は約5〜10秒かかります。")
    elif target_group == "🇺🇸 NASDAQ 100 (ハイテク代表銘柄)":
        ticker_list = get_ndx100()
        st.info(f"NASDAQ 100 構成銘柄 ({len(ticker_list)}銘柄) をスキャンします。")
    elif target_group == "🇺🇸 ダウ30種 (主要30銘柄)":
        ticker_list = get_dow30()
        st.info(f"ダウ工業株30種 ({len(ticker_list)}銘柄) をスキャンします。")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        scan_rsi_threshold = st.number_input('探索条件: RSI がこの数値以下であること', value=40, step=1, key='scan_rsi')
    with col2:
        scan_rsi_period = st.number_input('探索条件: RSI の計算期間', value=14, step=1, key='scan_rsi_p')

    # ニューヨーク時間に合わせる
    today_ny = datetime.datetime.now(ZoneInfo("America/New_York")).date()
    today_str = today_ny.strftime('%Y-%m-%d')

    if st.button('🚀 スクリーニング開始！', type='primary', key='scan_btn'):
        if not ticker_list:
            st.warning("銘柄リストが空です。")
        else:
            with st.spinner(f"ニューヨーク時間({today_str})のデータで戦略判定を実行中..."):
                # 超高速スクリーニング関数を呼び出す
                hit_tickers = run_fast_screening(
                    ticker_list=ticker_list, 
                    scan_rsi_threshold=scan_rsi_threshold, 
                    scan_rsi_period=scan_rsi_period, 
                    date_str=today_str
                )
                # スクリーニング実行状態と結果をセッションに保存
                st.session_state.hit_tickers = hit_tickers
                st.session_state.screening_run = True
                st.session_state.last_scan_date = today_str

    # -----------------------------------
    # 結果表示とタップ自動連携処理
    # -----------------------------------
    if st.session_state.screening_run:
        hit_tickers = st.session_state.hit_tickers
        if hit_tickers:
            st.success(f"🎉 厳しい条件をクリアした {len(hit_tickers)} 件のお宝銘柄が見つかりました！")
            st.info("💡 **テーブル内の銘柄行を直接タップ（クリック）** すると、サイドバーの「検証する銘柄コード」に自動入力されます。")
            
            df_hits = pd.DataFrame(hit_tickers)
            df_hits['最新株価($)'] = df_hits['最新株価($)'].round(2)
            df_hits['最新RSI'] = df_hits['最新RSI'].round(1)
            df_hits['MACD'] = df_hits['MACD'].round(3)
            df_hits['シグナル線'] = df_hits['シグナル線'].round(3)
            
            # テーブル（選択イベント付き）を描画
            event = st.dataframe(
                df_hits, 
                use_container_width=True,
                on_select="rerun", 
                selection_mode="single-row", # ← ★ココを single_row から single-row に修正しました！
                key="screening_results"
            )
            
            # 行がタップ選択された時の自動反映処理
            if event.selection.rows:
                selected_row = event.selection.rows[0]
                selected_ticker = df_hits.iloc[selected_row]['銘柄コード']
                
                # 新しく選択された銘柄が現在の検証用と異なれば、書き換えて画面を即リロード
                if st.session_state.calc_ticker != selected_ticker:
                    st.session_state.calc_ticker = selected_ticker
                    st.rerun()
            
            st.markdown(f"💡 **キャッシュの仕組み:** 米国東部時間（ニューヨーク）で日付が変わるまでは、再度ボタンを押しても一瞬でこの結果を再表示します。({st.session_state.last_scan_date}時点のデータ)")
        else:
            st.info("現在、指定された条件を満たす銘柄はありませんでした。RSIの基準値を少し上げてみるか、市場全体が下落している日を狙ってみてください。")
