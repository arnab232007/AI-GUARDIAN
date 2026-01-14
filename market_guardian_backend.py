import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# --- Configuration ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests for the frontend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GuardianBrain")

# --- AI Logic Core ---

class PatternRecognizer:
    """
    Mathematical definitions for candlestick and chart patterns.
    """
    @staticmethod
    def identify_candlesticks(df):
        """
        Adds boolean columns for candlestick patterns.
        """
        df = df.copy()
        o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
        
        # Candle Body Calculation
        body = np.abs(c - o)
        range_ = h - l
        
        # 1. Doji (Indecision)
        df['is_doji'] = body <= (range_ * 0.1)
        
        # 2. Hammer (Bullish Reversal)
        # Small body, long lower shadow, little/no upper shadow
        lower_shadow = np.minimum(o, c) - l
        df['is_hammer'] = (lower_shadow > (body * 2)) & (body < (range_ * 0.3))
        
        # 3. Bullish Engulfing (Strong Buy)
        # Prev candle red, curr candle green, curr body engulfs prev body
        prev_o, prev_c = o.shift(1), c.shift(1)
        df['is_engulfing'] = (prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c)
        
        # 4. Three White Soldiers (Strong Continuation)
        # 3 consecutive green candles, each closing higher
        df['is_3_soldiers'] = (
            (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) &
            (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        )
        
        return df

    @staticmethod
    def detect_chart_patterns(df):
        """
        Detects broader chart formations like Breakouts or Trends.
        """
        # Simple Breakout Logic: Price crosses 20-day High
        df['20_day_high'] = df['High'].rolling(window=20).max()
        df['is_breakout'] = df['Close'] > df['20_day_high'].shift(1)
        
        # Golden Cross (SMA 50 crosses SMA 200)
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        df['is_golden_cross'] = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
        
        return df

class MarketGuardian:
    def __init__(self):
        self.recognizer = PatternRecognizer()
    
    def fetch_live_data(self, symbol="BTC-USD", period="60d", interval="1h"):
        logger.info(f"Fetching live data for {symbol}...")
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty:
                return None
            # Ensure flat columns if MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return None

    def analyze_symbol(self, symbol):
        df = self.fetch_live_data(symbol)
        if df is None:
            return {"error": "Market data unavailable"}
        
        # 1. Pattern Recognition
        df = self.recognizer.identify_candlesticks(df)
        df = self.recognizer.detect_chart_patterns(df)
        
        latest = df.iloc[-1]
        
        # 2. Historical Cross-Check (Backtest)
        # Calculate how often the 'Engulfing' pattern resulted in a profit in the last 60 days
        success_rate = 0
        if latest['is_engulfing']:
            past_engulfing = df[df['is_engulfing']].copy()
            # Look forward 3 candles to see if price went up
            past_engulfing['profit'] = df['Close'].shift(-3) > df['Close']
            success_rate = past_engulfing['profit'].mean() if not past_engulfing.empty else 0

        # 3. Decision Logic
        score = 0
        reasons = []
        
        if latest['is_engulfing']: 
            score += 3
            reasons.append("Bullish Engulfing Pattern detected")
        if latest['is_hammer']:
            score += 1
            reasons.append("Hammer candle (Potential Reversal)")
        if latest['is_golden_cross']:
            score += 5
            reasons.append("Golden Cross (Long-term Bullish)")
        if latest['is_breakout']:
            score += 2
            reasons.append("Price broke 20-day resistance")
            
        # RSI Logic
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        if rsi < 30:
            score += 2
            reasons.append(f"RSI is Oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 2
            reasons.append(f"RSI is Overbought ({rsi:.1f})")

        # 4. Final Verdict
        action = "HOLD"
        color = "text-gray-400"
        if score >= 4:
            action = "STRONG BUY"
            color = "text-emerald-400"
        elif score >= 1:
            action = "BUY"
            color = "text-green-400"
        elif score <= -4:
            action = "STRONG SELL"
            color = "text-red-600"
        elif score <= -1:
            action = "SELL"
            color = "text-red-400"

        return {
            "symbol": symbol,
            "price": f"{latest['Close']:.2f}",
            "action": action,
            "confidence": f"{min(abs(score) * 20, 99):.0f}%",
            "reasons": reasons,
            "color": color,
            "historical_check": f"Historical accuracy for this pattern: {success_rate:.1%}" if success_rate > 0 else "Pattern rare in recent history."
        }

guardian = MarketGuardian()

# --- API Routes ---

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint for the frontend to call.
    Expects JSON: { "symbol": "BTC-USD" }
    """
    data = request.json
    symbol = data.get('symbol', 'BTC-USD')
    
    # Simulate AI "Thinking" time for UX
    time.sleep(1) 
    
    result = guardian.analyze_symbol(symbol)
    return jsonify(result)

@app.route('/news', methods=['GET'])
def news():
    """
    Returns simulated 'Real-time' news with sentiment analysis.
    In a real app, this would hit a News API.
    """
    headlines = [
        "Fed signals potential rate cuts in Q3.",
        "Bitcoin hash rate hits all-time high.",
        "Regulatory scrutiny increases for DeFi protocols.",
        "Tech sector earnings beat expectations."
    ]
    analyzed_news = []
    for h in headlines:
        sentiment = TextBlob(h).sentiment.polarity
        impact = "HIGH" if abs(sentiment) > 0.3 else "MED"
        analyzed_news.append({
            "title": h,
            "sentiment": "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral",
            "impact": impact,
            "time": datetime.now().strftime("%H:%M")
        })
    return jsonify(analyzed_news)

if __name__ == '__main__':
    print("--- MARKET GUARDIAN AI BACKEND RUNNING ON PORT 5000 ---")
    app.run(port=5000, debug=True)