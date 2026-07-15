import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. ページの基本設定
# ==========================================
st.set_page_config(page_title="トレード検証アプリ", layout="wide")
st.title('📈 トレード戦略 検証アプリ (MACD & RSI)')
st.markdown('銘柄と戦略を選んで、過去のデータに基づいた勝率とリターンを検証します。')

# ==========================================
# 2. サイドバー（入力フォーム）の設定
# ==========================================
st.sidebar.header('⚙️ 基本設定')
ticker = st.sidebar.text_input('銘柄コード (例: XOM, AAPL, 8306.T)', value='XOM')
days_later = st.sidebar.slider('何日後のリターンを計算する？', min_value=1, max_value=30, value=5)
period = st.sidebar.selectbox('データ取得期間', ['6mo', '1y', '2y', '5y', 'max'], index=1)

st.sidebar.header('🧠 戦略の選択')
strategy = st.sidebar.selectbox('検証するシグナル（買い条件）', [
    '1. MACD ゴールデンクロス', 
    '2. RSI 売られすぎからの反発', 
    '3. MACDクロス + RSI売られすぎ (合わせ技)'
])

st.sidebar.markdown('---')
st.sidebar.markdown('**パラメータ微調整**')
rsi_period = st.sidebar.number_input('RSI 期間', value=14, step=1)
rsi_threshold = st.sidebar.number_input('RSI 基準値 (この数値以下を売られすぎと判断)', value=40, step=1)

# ==========================================
# 3. メイン処理（ボタンが押されたら実行）
# ==========================================
if st.button('検証スタート！'):
    with st.spinner(f'{ticker} のデータを取得・計算中...'):
        
        # データの取得
        data = yf.Ticker(ticker).history(period=period)
        
        if data.empty:
            st.error("データの取得に失敗しました。銘柄コードを確認してください。")
        else:
            data.index = data.index.tz_localize(None) # タイムゾーン情報を削除
            
            # -----------------------------------
            # MACDの計算
            # -----------------------------------
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # -----------------------------------
            # RSIの計算
            # -----------------------------------
            delta = data['Close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/rsi_period, adjust=False).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # -----------------------------------
            # シグナルの判定（選んだ戦略によって条件を変える）
            # -----------------------------------
            # 条件1: MACDが下から上へ抜けた
            macd_cross = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
            # 条件2: RSIが基準値を下から上へ抜けた（反発）
            rsi_rebound = (data['RSI'] > rsi_threshold) & (data['RSI'].shift(1) <= rsi_threshold)
            # 条件3: MACDクロス発生時、同時にRSIが基準値以下であること（低い位置でのクロス）
            rsi_is_low = data['RSI'] <= rsi_threshold
            
            if '1. MACD' in strategy:
                data['Buy_Signal'] = macd_cross
            elif '2. RSI' in strategy:
                data['Buy_Signal'] = rsi_rebound
            elif '3. MACDクロス + RSI' in strategy:
                data['Buy_Signal'] = macd_cross & rsi_is_low
            
            # -----------------------------------
            # リターンの計算
            # -----------------------------------
            data['Price_Later'] = data['Close'].shift(-days_later)
            data['Return_(%)'] = ((data['Price_Later'] - data['Close']) / data['Close']) * 100
            
            signal_events = data[data['Buy_Signal']].copy()
            valid_returns = signal_events['Return_(%)'].dropna()
            
            # -----------------------------------
            # 結果の画面表示
            # -----------------------------------
            st.subheader(f'📊 検証結果: {ticker} (シグナル発生から {days_later} 営業日後)')
            st.markdown(f"**選択した戦略:** {strategy}")
            
            if len(valid_returns) > 0:
                win_rate = (valid_returns > 0).sum() / len(valid_returns) * 100
                avg_return = valid_returns.mean()
                
                # サマリー指標
                col1, col2, col3 = st.columns(3)
                col1.metric("シグナル発生回数", f"{len(valid_returns)} 回")
                col2.metric("勝率", f"{win_rate:.1f} %")
                col3.metric("平均リターン", f"{avg_return:+.2f} %")
                
                st.markdown("---")
                
                # グラフの描画（3段構成：株価 / MACD / RSI）
                st.markdown('**チャート推移とシグナル発生ポイント**')
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1, 1]})
                
                # 上段：株価
                ax1.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.7)
                signal_dates = signal_events.index
                signal_prices = signal_events['Close']
                ax1.scatter(signal_dates, signal_prices, marker='^', color='red', s=100, label='Buy Signal', zorder=5)
                ax1.set_ylabel('Price')
                ax1.legend()
                ax1.grid(True)
                
                # 中段：MACD
                ax2.plot(data.index, data['MACD'], label='MACD', color='blue')
                ax2.plot(data.index, data['Signal'], label='Signal', color='red', linestyle='--')
                ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax2.set_ylabel('MACD')
                ax2.legend(loc='upper left')
                ax2.grid(True)
                
                # 下段：RSI
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
                
                # 表の表示
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
