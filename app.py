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
    st.session_state.target_ticker = '8306.T' # 初期値を日本株(三菱UFJ)に変更
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

tab1, tab2 = st.tabs(["📊 過去の勝率検証 (単一銘柄)", "🔎 今日のチャンス探索 (大量スクリーニング)"])

# ==========================================
# 自動リスト取得用の関数
# ==========================================
@st.cache_data(ttl=86400)
def get_sp500():
    try:
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        df = pd.read_csv(url)
        return [str(t).replace('.', '-') for t in df['Symbol'].tolist()]
    except Exception:
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA']

@st.cache_data(ttl=86400)
def get_ndx100():
    return ['AAPL', 'ABNB', 'ADBE', 'AMD', 'AMZN', 'ASML', 'AVGO', 'COST', 'CRWD', 'CSCO', 
            'GOOG', 'GOOGL', 'INTC', 'META', 'MSFT', 'MU', 'NFLX', 'NVDA', 'PYPL', 'QCOM', 'TSLA', 'TXN']

@st.cache_data(ttl=86400)
def get_jp_popular():
    # スイングトレードで人気の日本株・中低位株・ETFの厳選リスト
    return ['1570.T', '1357.T', '5595.T', '8306.T', '8411.T', '8316.T', '9107.T', '9104.T', '9101.T', 
            '7203.T', '7201.T', '7267.T', '7011.T', '7012.T', '7013.T', '6526.T', '6920.T', '8035.T', 
            '9984.T', '6758.T', '1605.T', '5401.T', '4755.T', '9501.T', '9432.T', '9433.T', '9434.T', 
            '3402.T', '4005.T', '6501.T', '8001.T', '8058.T', '8031.T', '3382.T', '4502.T', '6902.T']

@st.cache_data(ttl=86400)
def get_jp_core30():
    # TOPIX Core30 (日本の超大型株)
    return ['7203.T', '8306.T', '9984.T', '6861.T', '9432.T', '6758.T', '8035.T', '9983.T', '8316.T', 
            '8058.T', '4063.T', '6920.T', '6501.T', '7974.T', '8001.T', '4568.T', '8031.T', '8766.T', 
            '8411.T', '6098.T', '4502.T', '7741.T', '4519.T', '6902.T', '3382.T', '9433.T', '6367.T', 
            '6594.T', '8053.T', '5108.T']

# ==========================================
# ★超高速一括スクリーニング関数
# ==========================================
@st.cache_data(ttl=86400)
def run_fast_screening(ticker_list, scan_rsi_threshold, scan_rsi_period, date_str, use_trend_filter, filter_ma_period, use_price_filter, max_price_limit):
    hit_tickers = []
    download_period = '1y' if (use_trend_filter and filter_ma_period >= 100) else '6mo'
    
    raw_data = yf.download(
        ticker_list, 
        period=download_period, 
        group_by='ticker', 
        threads=True, 
        progress=False, 
        auto_adjust=True
    )
    
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
            
            close_prices = scan_data['Close']
            
            # 株価上限フィルターの適用
            latest_price = float(close_prices.iloc[-1])
            if use_price_filter and latest_price > max_price_limit:
                continue
            
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            delta = close_prices.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/scan_rsi_period, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/scan_rsi_period, adjust=False).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if use_trend_filter:
                scan_data['MA'] = close_prices.rolling(window=filter_ma_period).mean()
            
            latest_idx = -1
            prev_idx = -2
            
            latest_rsi = rsi.iloc[latest_idx]
            latest_macd = macd.iloc[latest_idx]
            latest_sig = signal.iloc[latest_idx]
            
            prev_macd = macd.iloc[prev_idx]
            prev_sig = signal.iloc[prev_idx]
            
            cond_rsi = latest_rsi <= scan_rsi_threshold
            
            cond_trend = True
            if use_trend_filter:
                if 'MA' in scan_data.columns and not pd.isna(scan_data['MA'].iloc[latest_idx]):
                    cond_trend = latest_price > scan_data['MA'].iloc[latest_idx]
                else:
                    cond_trend = False
            
            cond_under = latest_macd < latest_sig
            diff_latest = latest_sig - latest_macd
            diff_prev = prev_sig - prev_macd
            cond_closing = diff_latest < diff_prev
            cond_macd_up = latest_macd > prev_macd
            is_approaching = cond_under and cond_closing and cond_macd_up
            
            is_crossed = (latest_macd > latest_sig) and (prev_macd <= prev_sig)
            
            if cond_rsi and cond_trend and (is_approaching or is_crossed):
                status = "🟢 クロス直後!" if is_crossed else "🟡 クロス直前"
                res_dict = {
                    '銘柄コード': t,
                    '状態': status,
                    '最新株価': latest_price,
                    '最新RSI': float(latest_rsi),
                    'MACD': float(latest_macd),
                    'シグナル線': float(latest_sig)
                }
                if use_trend_filter:
                    res_dict[f'MA({filter_ma_period})'] = float(scan_data['MA'].iloc[latest_idx].round(2))
                hit_tickers.append(res_dict)
        except Exception:
            pass
            
    return hit_tickers

# ==========================================
# 2. サイドバー（検証タブ用の設定）
# ==========================================
st.sidebar.header('⚙️ 検証タブ用 基本設定')
input_val = st.sidebar.text_input('検証する銘柄コード (日本株は末尾に.T)', value=st.session_state.target_ticker)
st.session_state.target_ticker = input_val
ticker = st.session_state.target_ticker

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
st.sidebar.markdown('**📊 移動平均線 (MA) 設定**')
show_ma = st.sidebar.checkbox('検証チャートにMAを表示', value=True)
ma_period = st.sidebar.number_input('検証用 MA 期間', value=75, min_value=5, max_value=200, step=1)

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
            
            # 日本株か米国株かでタイムゾーンを自動判定
            tz_str = "Asia/Tokyo" if ".T" in ticker.upper() else "America/New_York"
            today_date = datetime.datetime.now(ZoneInfo(tz_str)).date()
            
            if period == 'max':
                data = yf.Ticker(ticker).history(period='max')
                valid_start_date = None
            else:
                warmup_days = int(ma_period + 30)
                if period.endswith('mo'):
                    months = int(period.replace('mo', ''))
                    start_date = today_date - pd.DateOffset(months=months) - pd.DateOffset(days=warmup_days)
                    valid_start_date = today_date - pd.DateOffset(months=months)
                elif period.endswith('y'):
                    years = int(period.replace('y', ''))
                    start_date = today_date - pd.DateOffset(years=years) - pd.DateOffset(days=warmup_days)
                    valid_start_date = today_date - pd.DateOffset(years=years)
                
                data = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'))
            
            if data.empty:
                st.error("データの取得に失敗しました。日本株の場合は末尾に「.T」をつけてください（例：8306.T）。")
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
                
                data['MA'] = data['Close'].rolling(window=ma_period).mean()
                
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
                
                if valid_start_date is not None:
                    data = data[data.index >= pd.to_datetime(valid_start_date)]
                
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
                    if show_ma:
                        ax1.plot(data.index, data['MA'], label=f'{ma_period} MA', color='orange', linestyle='--', alpha=0.8)
                    
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
    st.markdown('### 🔎 日米株 大量スクリーニング (MACDクロス直前/直後 × RSI売られすぎ)')
    
    target_group = st.radio(
        "スクリーニング対象のリストを選択してください",
        ("🇯🇵 日本株 (人気銘柄・ETF)", "🇯🇵 TOPIX Core30 (大型株)", "🇺🇸 S&P 500 (米国大型)", "🇺🇸 NASDAQ 100", "✍️ 自分で入力する"),
        horizontal=True
    )
    
    ticker_list = []
    if target_group == "✍️ 自分で入力する":
        default_tickers = '8306.T, 5595.T, 1570.T, 7203.T'
        tickers_input = st.text_area('監視する銘柄コード (カンマ区切り。日本株は末尾に .T をつける)', default_tickers)
        ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    elif target_group == "🇯🇵 日本株 (人気銘柄・ETF)":
        ticker_list = get_jp_popular()
        st.info(f"値動きの活発な日本の人気銘柄とETF ({len(ticker_list)}銘柄) をスキャンします。")
    elif target_group == "🇯🇵 TOPIX Core30 (大型株)":
        ticker_list = get_jp_core30()
        st.info(f"日本の超大型株 ({len(ticker_list)}銘柄) をスキャンします。")
    elif target_group == "🇺🇸 S&P 500 (米国大型)":
        ticker_list = get_sp500()
        st.info(f"S&P 500 構成銘柄 ({len(ticker_list)}銘柄) をスキャンします。")
    elif target_group == "🇺🇸 NASDAQ 100":
        ticker_list = get_ndx100()
        st.info(f"NASDAQ 100 構成銘柄 ({len(ticker_list)}銘柄) をスキャンします。")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        scan_rsi_threshold = st.number_input('探索条件: RSI がこの数値以下であること', value=40, step=1, key='scan_rsi')
    with col2:
        scan_rsi_period = st.number_input('探索条件: RSI の計算期間', value=14, step=1, key='scan_rsi_p')

    st.markdown('**💰 資金フィルター (予算に合わせた銘柄選び)**')
    use_price_filter = st.checkbox('指定した株価以下の銘柄のみに絞り込む', value=True)
    if use_price_filter:
        max_price_limit = st.number_input('上限株価 (日本株は「円」、米国株は「ドル」)', value=2000.0, step=100.0)
    else:
        max_price_limit = 9999999.0

    st.markdown('**📈 トレードフィルター (ダマシ回避用)**')
    use_trend_filter = st.checkbox('長期移動平均線 (MA) より上に位置する銘柄のみに絞り込む (上昇トレンド優先)', value=False)
    if use_trend_filter:
        filter_ma_period = st.number_input('フィルターに使用する MA 期間 (例: 75, 200)', value=75, min_value=5, max_value=200, step=1)
    else:
        filter_ma_period = 75

    tz_str = "Asia/Tokyo" if "🇯🇵" in target_group or ".T" in "".join(ticker_list) else "America/New_York"
    today_ny = datetime.datetime.now(ZoneInfo(tz_str)).date()
    today_str = today_ny.strftime('%Y-%m-%d')

    if st.button('🚀 スクリーニング開始！', type='primary', key='scan_btn'):
        if not ticker_list:
            st.warning("銘柄リストが空です。")
        else:
            with st.spinner(f"基準日({today_str})のデータで戦略判定を実行中..."):
                hit_tickers = run_fast_screening(
                    ticker_list=ticker_list, 
                    scan_rsi_threshold=scan_rsi_threshold, 
                    scan_rsi_period=scan_rsi_period, 
                    date_str=today_str,
                    use_trend_filter=use_trend_filter,
                    filter_ma_period=filter_ma_period,
                    use_price_filter=use_price_filter,
                    max_price_limit=max_price_limit
                )
                st.session_state.hit_tickers = hit_tickers
                st.session_state.screening_run = True
                st.session_state.last_scan_date = today_str

    if st.session_state.screening_run:
        hit_tickers = st.session_state.hit_tickers
        if hit_tickers:
            st.success(f"🎉 条件をクリアした {len(hit_tickers)} 件のお宝銘柄が見つかりました！")
            st.info("💡 **テーブル内の銘柄行を直接タップ（クリック）** すると、サイドバーの「検証する銘柄コード」に自動入力されます。")
            
            df_hits = pd.DataFrame(hit_tickers)
            df_hits['最新株価'] = df_hits['最新株価'].round(2)
            df_hits['最新RSI'] = df_hits['最新RSI'].round(1)
            df_hits['MACD'] = df_hits['MACD'].round(3)
            df_hits['シグナル線'] = df_hits['シグナル線'].round(3)
            
            df_crossed = df_hits[df_hits['状態'] == "🟢 クロス直後!"].reset_index(drop=True)
            df_approaching = df_hits[df_hits['状態'] == "🟡 クロス直前"].reset_index(drop=True)
            
            st.subheader(f"🟢 ゴールデンクロスした直後の銘柄 ({len(df_crossed)}件)")
            if not df_crossed.empty:
                event_crossed = st.dataframe(
                    df_crossed.drop(columns=['状態']), 
                    use_container_width=True,
                    on_select="rerun", 
                    selection_mode="single-row",
                    key="screening_results_crossed"
                )
                if event_crossed.selection.rows:
                    selected_row = event_crossed.selection.rows[0]
                    selected_ticker = df_crossed.iloc[selected_row]['銘柄コード']
                    if st.session_state.target_ticker != selected_ticker:
                        st.session_state.target_ticker = selected_ticker
                        st.rerun()
            else:
                st.write("現在、該当する銘柄はありません。")

            st.markdown("---")

            st.subheader(f"🟡 ゴールデンクロスしそうな銘柄 ({len(df_approaching)}件)")
            if not df_approaching.empty:
                event_approaching = st.dataframe(
                    df_approaching.drop(columns=['状態']), 
                    use_container_width=True,
                    on_select="rerun", 
                    selection_mode="single-row",
                    key="screening_results_approaching"
                )
                if event_approaching.selection.rows:
                    selected_row = event_approaching.selection.rows[0]
                    selected_ticker = df_approaching.iloc[selected_row]['銘柄コード']
                    if st.session_state.target_ticker != selected_ticker:
                        st.session_state.target_ticker = selected_ticker
                        st.rerun()
            else:
                st.write("現在、該当する銘柄はありません。")
            
            st.markdown("---")
            st.markdown(f"💡 **キャッシュの仕組み:** 対象市場が閉まるまでは、再度ボタンを押しても一瞬でこの結果を再表示します。({st.session_state.last_scan_date}時点のデータ)")
        else:
            st.info("現在、指定された条件を満たす銘柄はありませんでした。RSIの基準値を上げるか、資金フィルターの上限を引き上げてみてください。")
