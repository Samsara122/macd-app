import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. ページの基本設定
# ==========================================
st.set_page_config(page_title="トレード検証＆探索アプリ", layout="wide")
st.title('📈 トレード戦略 検証＆探索アプリ (MACD & RSI)')

# タブの作成
tab1, tab2 = st.tabs(["📊 過去の勝率検証 (単一銘柄)", "🔎 今日のチャンス探索 (複数銘柄スクリーニング)"])

# ==========================================
# 2. サイドバー（検証タブ用の設定）
# ==========================================
st.sidebar.header('⚙️ 検証タブ用 基本設定')
ticker = st.sidebar.text_input('検証する銘柄コード (例: XOM, AAPL, 8306.T)', value='XOM')
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
                
                # MACDの計算
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                
                # RSIの計算
                delta = data['Close'].diff()
                gain = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
                loss = -delta.clip(upper=0).ewm(alpha=1/rsi_period, adjust=False).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                # シグナル判定
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
                
                # リターンの計算
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
    st.markdown('### 🔎 「もうすぐMACDがゴールデンクロスしそう」な銘柄を探す')
    st.markdown('指定した複数の銘柄の中から、**RSIが一定以下** かつ **MACDとシグナル線の差が縮まってクロス直前** の銘柄をリストアップします。')
    
    # デフォルトでいくつかの有名銘柄を入れておく
    default_tickers = 'AAPL, MSFT, GOOG, AMZN, NVDA, TSLA, META, XOM, JNJ, JPM, 7203.T, 8306.T, 9984.T'
    tickers_input = st.text_area('監視する銘柄コード (カンマ区切りで複数入力)', default_tickers)
    
    col1, col2 = st.columns(2)
    with col1:
        scan_rsi_threshold = st.number_input('探索条件: RSI がこの数値以下であること', value=45, step=1, key='scan_rsi')
    with col2:
        scan_rsi_period = st.number_input('探索条件: RSI の計算期間', value=14, step=1, key='scan_rsi_p')

    if st.button('スクリーニング開始！', key='scan_btn'):
        ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
        
        if not ticker_list:
            st.warning("銘柄コードを入力してください。")
        else:
            hit_tickers = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, t in enumerate(ticker_list):
                status_text.text(f"データ取得・解析中... {t} ({i+1}/{len(ticker_list)})")
                try:
                    # MACD計算のため過去3ヶ月分のデータを取得
                    scan_data = yf.Ticker(t).history(period='3mo')
                    if scan_data.empty or len(scan_data) < 30:
                        continue
                        
                    scan_data.index = scan_data.index.tz_localize(None)
                    
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
                    
                    # 最新日と前日のデータを取得
                    latest = scan_data.iloc[-1]
                    prev = scan_data.iloc[-2]
                    
                    # --- 「ゴールデンクロスしそう」の条件判定 ---
                    
                    # 条件1. RSIが基準値以下
                    cond_rsi = latest['RSI'] <= scan_rsi_threshold
                    
                    # 条件2. MACDがまだシグナルの下にある（未クロス）
                    cond_under = latest['MACD'] < latest['Signal']
                    
                    # 条件3. MACDとシグナルの差が前日より縮まっている（クロスに向かっている）
                    diff_latest = latest['Signal'] - latest['MACD']
                    diff_prev = prev['Signal'] - prev['MACD']
                    cond_closing = diff_latest < diff_prev
                    
                    # 条件4. MACD自体が上向いている
                    cond_macd_up = latest['MACD'] > prev['MACD']
                    
                    # すべての条件を満たしたらリストに追加
                    if cond_rsi and cond_under and cond_closing and cond_macd_up:
                        hit_tickers.append({
                            '銘柄コード': t,
                            '最新株価': latest['Close'],
                            '最新RSI': latest['RSI'],
                            'MACD': latest['MACD'],
                            'シグナル線': latest['Signal']
                        })
                except Exception as e:
                    pass
                
                progress_bar.progress((i + 1) / len(ticker_list))
                
            status_text.text("スクリーニング完了！")
            
            if hit_tickers:
                st.success(f"🎉 {len(hit_tickers)} 件の「ゴールデンクロス直前」の銘柄が見つかりました！")
                
                # 見やすく整形して表で表示
                df_hits = pd.DataFrame(hit_tickers)
                df_hits['最新株価'] = df_hits['最新株価'].round(2)
                df_hits['最新RSI'] = df_hits['最新RSI'].round(1)
                df_hits['MACD'] = df_hits['MACD'].round(3)
                df_hits['シグナル線'] = df_hits['シグナル線'].round(3)
                
                st.dataframe(df_hits, use_container_width=True)
                
                st.markdown("💡 **次のアクション:** 上記の銘柄を実際のチャートで確認し、仕込みのタイミングを探ってみてください！")
            else:
                st.info("現在、指定された条件（RSIが低く、MACDがクロス直前）を満たす銘柄はありませんでした。")

