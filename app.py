import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. ページの基本設定
# ==========================================
st.set_page_config(page_title="MACD バックテストアプリ", layout="wide")
st.title('📈 MACD ゴールデンクロス検証アプリ')
st.markdown('銘柄コードと日数を指定して、MACDゴールデンクロスの有効性を検証します。')

# ==========================================
# 2. サイドバー（入力フォーム）の設定
# ==========================================
st.sidebar.header('⚙️ 検索条件の設定')

# 銘柄コードの入力（テキストボックス）
ticker = st.sidebar.text_input('銘柄コード (例: XOM, AAPL, 8306.T)', value='XOM')

# 何日後かの入力（スライダーで1日〜30日まで直感的に選べるようにする）
days_later = st.sidebar.slider('何日後のリターンを計算する？', min_value=1, max_value=30, value=5)

# データ取得期間の選択（セレクトボックス）
period = st.sidebar.selectbox('データ取得期間', ['6mo', '1y', '2y', '5y', 'max'], index=1)

# ==========================================
# 3. メイン処理（ボタンが押されたら実行）
# ==========================================
if st.button('検証スタート！'):
    # 読み込み中のくるくるアニメーションを表示
    with st.spinner(f'{ticker} のデータを取得・計算中...'):
        
        # データの取得
        data = yf.Ticker(ticker).history(period=period)
        
        if data.empty:
            st.error("データの取得に失敗しました。銘柄コードが正しいか確認してください。")
        else:
            data.index = data.index.tz_localize(None) # タイムゾーン情報を削除
            
            # MACDとシグナルの計算
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # ゴールデンクロスの判定
            data['Golden_Cross'] = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
            
            # リターンの計算
            data['Price_Later'] = data['Close'].shift(-days_later)
            data['Return_(%)'] = ((data['Price_Later'] - data['Close']) / data['Close']) * 100
            
            cross_events = data[data['Golden_Cross']].copy()
            valid_returns = cross_events['Return_(%)'].dropna()
            
            # -----------------------------------
            # 結果の画面表示
            # -----------------------------------
            st.subheader(f'📊 検証結果: {ticker} (シグナル発生から {days_later} 営業日後)')
            
            if len(valid_returns) > 0:
                win_rate = (valid_returns > 0).sum() / len(valid_returns) * 100
                avg_return = valid_returns.mean()
                
                # サマリー指標を3つ並べてかっこよく表示
                col1, col2, col3 = st.columns(3)
                col1.metric("総クロス回数", f"{len(valid_returns)} 回")
                col2.metric("勝率", f"{win_rate:.1f} %")
                col3.metric("平均リターン", f"{avg_return:+.2f} %")
                
                st.markdown("---")
                
                # グラフの描画（文字化け対策でグラフ内は英語）
                st.markdown('**株価とMACDの推移チャート**')
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
                
                ax1.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.7)
                cross_dates = cross_events.index
                cross_prices = cross_events['Close']
                ax1.scatter(cross_dates, cross_prices, marker='^', color='red', s=100, label='Golden Cross (Buy)', zorder=5)
                ax1.set_ylabel('Price')
                ax1.legend()
                ax1.grid(True)
                
                ax2.plot(data.index, data['MACD'], label='MACD', color='blue')
                ax2.plot(data.index, data['Signal'], label='Signal', color='red', linestyle='--')
                ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('MACD')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                
                # MatplotlibのグラフをStreamlitの画面上に表示
                st.pyplot(fig)
                
                st.markdown("---")
                
                # 表の表示
                st.markdown('**詳細データ一覧**')
                display_df = pd.DataFrame({
                    '購入時の株価': cross_events['Close'].round(2),
                    f'{days_later}日後の株価': cross_events['Price_Later'].round(2),
                    'リターン(%)': cross_events['Return_(%)'].round(2)
                })
                display_df.index = display_df.index.strftime('%Y-%m-%d')
                
                # 表を画面上に綺麗に表示
                st.dataframe(display_df, use_container_width=True)
                
            else:
                st.warning("指定された期間内にゴールデンクロスが発生していないか、検証データが不足しています。")