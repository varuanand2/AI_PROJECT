# backend.py
# Run: python backend.py
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import asyncio
import time

import requests
import aiohttp
import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
import talib

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 @app.get("/")
def read_root():
    return {"message": "Hello World! My API is running."}

# ================= UPSTOX CONFIG ==================

UPSTOX_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"  # yahan apna token daalo

class UpstoxDataFetcher:
    BASE_URL = "https://api.upstox.com/v2"

    def __init__(self, access_token: str):
        self.access_token = access_token

    def _headers(self):
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def get_live_ltp(self, instrument_key: str) -> float:
        """
        Upstox LTP API.
        Ek ya multiple instrument_key de sakte ho, comma-separated. [web:9]
        """
        try:
            url = f"{self.BASE_URL}/market-quote/ltp?instrument_key={instrument_key}"
            resp = requests.get(url, headers=self._headers(), timeout=5)
            if resp.status_code != 200:
                logger.error(f"LTP error: {resp.status_code} {resp.text}")
                return 0.0
            data = resp.json().get("data", {})
            if instrument_key not in data:
                return 0.0
            return float(data[instrument_key]["last_price"])
        except Exception as e:
            logger.error(f"get_live_ltp exception: {e}")
            return 0.0

    def get_option_ltp(
        self, underlying_instrument_key: str, expiry: str, strike: float, option_type: str
    ) -> float:
        """
        Yahaan tum apni mapping lagaoge:
        - Upstox instruments JSON / BOD file se instrument_key nikalna hoga. [web:2]
        - Abhi ke liye dummy: directly LTP ek given key se de rahe hain.
        """
        # TODO: expiry, strike, option_type -> instrument_key mapping banaani hai.
        # Placeholder: directly underlying_instrument_key se LTP de raha hai.
        return self.get_live_ltp(underlying_instrument_key)


# ============== ADVANCED ANALYZER (yfinance + TA-Lib) ==============

@dataclass
class NewsArticle:
    title: str
    summary: str
    url: str
    source: str
    published_date: datetime
    category: str = "general"
    symbol: Optional[str] = None


class AdvancedStockAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.stock_symbols = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'HDFC BANK': 'HDFCBANK.NS',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFY': 'INFY.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'BAJAJ FINANCE': 'BAJFINANCE.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'ITC': 'ITC.NS',
            'KOTAK BANK': 'KOTAKBANK.NS',
            'WIPRO': 'WIPRO.NS',
            'MARUTI': 'MARUTI.NS',
            'HCLTECH': 'HCLTECH.NS',
            'L&T': 'LT.NS',
            'AXIS BANK': 'AXISBANK.NS'
        }

    def clean_value(self, val):
        if val is None:
            return None
        if isinstance(val, (float, np.floating)):
            if pd.isna(val) or np.isnan(val) or np.isinf(val):
                return None
        return float(val) if isinstance(val, (np.floating, np.integer)) else val

    def get_chart_data(self, symbol, period='1y', interval='1d'):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                return []
            candles = []
            for idx, row in data.iterrows():
                candles.append({
                    'time': int(idx.timestamp()),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                })
            return candles
        except Exception as e:
            logger.error(f"Error fetching chart data for {symbol}: {e}")
            return []

    def get_stock_data_multiple_timeframes(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            data_1h = ticker.history(period='15d', interval='1h')
            data_1d = ticker.history(period='14d', interval='1d')
            data_long = ticker.history(period='10y')
            info = ticker.info
            if data_1h.empty or data_1d.empty:
                logger.warning(f"No data found for {symbol}")
                return None, None, None, None
            return data_1h, data_1d, data_long, info
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None, None, None

    def calculate_fibonacci_1h_advanced(self, data_1h):
        try:
            swing_high = float(data_1h['High'].max())
            swing_low = float(data_1h['Low'].min())
            diff = swing_high - swing_low

            fib_levels = {
                '0.0': round(swing_high, 2),
                '23.6': round(swing_high - (diff * 0.236), 2),
                '38.2': round(swing_high - (diff * 0.382), 2),
                '50.0': round(swing_high - (diff * 0.500), 2),
                '61.8': round(swing_high - (diff * 0.618), 2),
                '78.6': round(swing_high - (diff * 0.786), 2),
                '100.0': round(swing_low, 2)
            }

            importance_analysis = []

            for level_name, level_price in fib_levels.items():
                touch_count = 0
                support_count = 0
                resistance_count = 0
                tolerance = level_price * 0.005

                for idx in range(1, len(data_1h)):
                    high = data_1h['High'].iloc[idx]
                    low = data_1h['Low'].iloc[idx]
                    close = data_1h['Close'].iloc[idx]
                    prev_close = data_1h['Close'].iloc[idx - 1]

                    if low <= level_price + tolerance and high >= level_price - tolerance:
                        touch_count += 1
                        if prev_close < level_price and close > level_price:
                            support_count += 1
                        elif prev_close > level_price and close < level_price:
                            resistance_count += 1

                if support_count > resistance_count:
                    level_type = "Support"
                elif resistance_count > support_count:
                    level_type = "Resistance"
                else:
                    level_type = "Neutral"

                if touch_count >= 5:
                    importance = "Very Strong"
                elif touch_count >= 3:
                    importance = "Strong"
                elif touch_count >= 1:
                    importance = "Moderate"
                else:
                    importance = "Weak"

                importance_analysis.append({
                    'level': level_name,
                    'price': level_price,
                    'touches': touch_count,
                    'type': level_type,
                    'importance': importance,
                    'support_reactions': support_count,
                    'resistance_reactions': resistance_count
                })

            importance_analysis.sort(key=lambda x: x['touches'], reverse=True)

            return {
                'timeframe': '1H',
                'period': '10-15 days',
                'swing_high': round(swing_high, 2),
                'swing_low': round(swing_low, 2),
                'difference': round(diff, 2),
                'levels': fib_levels,
                'importance_analysis': importance_analysis,
                'most_important_levels': importance_analysis[:3]
            }

        except Exception as e:
            logger.error(f"Fibonacci 1H calculation error: {e}")
            return {}

    def calculate_support_resistance_1d_advanced(self, data_1d, current_price):
        try:
            if data_1d is None or len(data_1d) < 2:
                logger.warning("Insufficient data for S/R analysis")
                return self._get_default_sr_structure(current_price)

            data_2weeks = data_1d.tail(14)
            price_levels = []

            for idx in range(len(data_2weeks)):
                high = data_2weeks['High'].iloc[idx]
                low = data_2weeks['Low'].iloc[idx]
                volume = data_2weeks['Volume'].iloc[idx]

                price_levels.append({
                    'price': round(high, 2),
                    'type': 'resistance_candidate',
                    'volume': volume,
                    'date': data_2weeks.index[idx]
                })

                price_levels.append({
                    'price': round(low, 2),
                    'type': 'support_candidate',
                    'volume': volume,
                    'date': data_2weeks.index[idx]
                })

            def cluster_prices(prices, tolerance=0.01):
                if not prices:
                    return []
                prices = sorted(prices, key=lambda x: x['price'])
                clusters = []
                current_cluster = [prices[0]]
                for i in range(1, len(prices)):
                    if abs(prices[i]['price'] - current_cluster[0]['price']) / current_cluster[0]['price'] <= tolerance:
                        current_cluster.append(prices[i])
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [prices[i]]
                clusters.append(current_cluster)
                return clusters

            support_candidates = [p for p in price_levels if p['type'] == 'support_candidate']
            resistance_candidates = [p for p in price_levels if p['type'] == 'resistance_candidate']

            support_clusters = cluster_prices(support_candidates)
            resistance_clusters = cluster_prices(resistance_candidates)

            def analyze_cluster(cluster, cluster_type):
                avg_price = sum(p['price'] for p in cluster) / len(cluster)
                total_volume = sum(p['volume'] for p in cluster)
                touch_count = len(cluster)

                buy_count = 0
                sell_count = 0

                for idx in range(1, len(data_2weeks)):
                    high = data_2weeks['High'].iloc[idx]
                    low = data_2weeks['Low'].iloc[idx]
                    close = data_2weeks['Close'].iloc[idx]
                    open_price = data_2weeks['Open'].iloc[idx]
                    prev_close = data_2weeks['Close'].iloc[idx - 1]

                    tolerance = avg_price * 0.005

                    if low <= avg_price + tolerance and high >= avg_price - tolerance:
                        if close > open_price and close > prev_close:
                            buy_count += 1
                        elif close < open_price and close < prev_close:
                            sell_count += 1

                if touch_count >= 4:
                    strength = "Very Strong"
                elif touch_count >= 3:
                    strength = "Strong"
                elif touch_count >= 2:
                    strength = "Moderate"
                else:
                    strength = "Weak"

                return {
                    'price': round(avg_price, 2),
                    'touches': touch_count,
                    'buy_reactions': buy_count,
                    'sell_reactions': sell_count,
                    'total_volume': int(total_volume),
                    'strength': strength,
                    'type': cluster_type,
                    'last_tested': cluster[-1]['date'].strftime('%Y-%m-%d')
                }

            support_levels = []
            for cluster in support_clusters:
                if cluster:
                    analysis = analyze_cluster(cluster, 'support')
                    if analysis['price'] < current_price:
                        support_levels.append(analysis)

            resistance_levels = []
            for cluster in resistance_clusters:
                if cluster:
                    analysis = analyze_cluster(cluster, 'resistance')
                    if analysis['price'] > current_price:
                        resistance_levels.append(analysis)

            support_levels.sort(key=lambda x: (x['touches'], x['total_volume']), reverse=True)
            resistance_levels.sort(key=lambda x: (x['touches'], x['total_volume']), reverse=True)

            support_levels = support_levels[:5] if support_levels else []
            resistance_levels = resistance_levels[:5] if resistance_levels else []

            if not support_levels:
                support_levels = [{
                    'price': round(current_price * 0.98, 2),
                    'touches': 1,
                    'buy_reactions': 0,
                    'sell_reactions': 0,
                    'total_volume': 0,
                    'strength': 'Weak',
                    'type': 'support',
                    'last_tested': datetime.now().strftime('%Y-%m-%d')
                }]

            if not resistance_levels:
                resistance_levels = [{
                    'price': round(current_price * 1.02, 2),
                    'touches': 1,
                    'buy_reactions': 0,
                    'sell_reactions': 0,
                    'total_volume': 0,
                    'strength': 'Weak',
                    'type': 'resistance',
                    'last_tested': datetime.now().strftime('%Y-%m-%d')
                }]

            all_levels = support_levels + resistance_levels
            all_levels.sort(key=lambda x: (x['buy_reactions'] + x['sell_reactions']), reverse=True)
            most_active_level = all_levels[0] if all_levels else None

            return {
                'timeframe': '1D',
                'period': '2 weeks (14 days)',
                'current_price': round(current_price, 2),
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'most_active_level': most_active_level,
                'summary': {
                    'total_support_levels': len(support_levels),
                    'total_resistance_levels': len(resistance_levels),
                    'strongest_support': support_levels[0]['price'] if support_levels else None,
                    'strongest_resistance': resistance_levels[0]['price'] if resistance_levels else None
                }
            }

        except Exception as e:
            logger.error(f"Support/Resistance 1D calculation error: {e}")
            return self._get_default_sr_structure(current_price)

    def _get_default_sr_structure(self, current_price):
        return {
            'timeframe': '1D',
            'period': '2 weeks (14 days)',
            'current_price': round(current_price, 2),
            'support_levels': [{
                'price': round(current_price * 0.98, 2),
                'touches': 0,
                'buy_reactions': 0,
                'sell_reactions': 0,
                'total_volume': 0,
                'strength': 'Unknown',
                'type': 'support',
                'last_tested': datetime.now().strftime('%Y-%m-%d')
            }],
            'resistance_levels': [{
                'price': round(current_price * 1.02, 2),
                'touches': 0,
                'buy_reactions': 0,
                'sell_reactions': 0,
                'total_volume': 0,
                'strength': 'Unknown',
                'type': 'resistance',
                'last_tested': datetime.now().strftime('%Y-%m-%d')
            }],
            'most_active_level': None,
            'summary': {
                'total_support_levels': 1,
                'total_resistance_levels': 1,
                'strongest_support': round(current_price * 0.98, 2),
                'strongest_resistance': round(current_price * 1.02, 2)
            }
        }

    def calculate_vwap(self, data):
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            current_vwap = self.clean_value(vwap.iloc[-1])
            return {
                'current_vwap': current_vwap if current_vwap else None,
                'vwap_series': vwap
            }
        except Exception as e:
            logger.error(f"VWAP calculation error: {e}")
            return {'current_vwap': None, 'vwap_series': None}

    def calculate_supertrend(self, data, period=10, multiplier=3):
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']

            atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            hl_avg = (high + low) / 2
            upper_band = hl_avg + (multiplier * atr)
            lower_band = hl_avg - (multiplier * atr)

            supertrend = pd.Series(index=data.index, dtype=float)
            direction = pd.Series(index=data.index, dtype=int)

            supertrend.iloc[0] = lower_band.iloc[0]
            direction.iloc[0] = 1

            for i in range(1, len(data)):
                if close.iloc[i] > supertrend.iloc[i - 1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                elif close.iloc[i] < supertrend.iloc[i - 1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = supertrend.iloc[i - 1]
                    direction.iloc[i] = direction.iloc[i - 1]

                    if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend.iloc[i - 1]:
                        supertrend.iloc[i] = supertrend.iloc[i - 1]
                    elif direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend.iloc[i - 1]:
                        supertrend.iloc[i] = supertrend.iloc[i - 1]

            current_supertrend = self.clean_value(supertrend.iloc[-1])
            current_direction = int(direction.iloc[-1])
            signal = "BUY" if current_direction == 1 else "SELL"

            return {
                'supertrend_value': current_supertrend,
                'signal': signal,
                'direction': current_direction
            }
        except Exception as e:
            logger.error(f"Supertrend calculation error: {e}")
            return {'supertrend_value': None, 'signal': 'HOLD', 'direction': 0}

    def detect_candlestick_patterns(self, data):
        patterns_detected = []
        try:
            o = data['Open'].values
            h = data['High'].values
            l = data['Low'].values
            c = data['Close'].values

            pattern_funcs = {
                'Doji': talib.CDLDOJI,
                'Hammer': talib.CDLHAMMER,
                'Shooting Star': talib.CDLSHOOTINGSTAR,
                'Engulfing': talib.CDLENGULFING,
                'Morning Star': talib.CDLMORNINGSTAR,
                'Evening Star': talib.CDLEVENINGSTAR,
                'Three White Soldiers': talib.CDL3WHITESOLDIERS,
                'Three Black Crows': talib.CDL3BLACKCROWS,
                'Hanging Man': talib.CDLHANGINGMAN,
                'Inverted Hammer': talib.CDLINVERTEDHAMMER
            }

            for name, func in pattern_funcs.items():
                result = func(o, h, l, c)
                if result[-1] != 0:
                    signal = "Bullish" if result[-1] > 0 else "Bearish"
                    patterns_detected.append(f"{name} ({signal})")

        except Exception as e:
            logger.error(f"Candlestick pattern detection error: {e}")

        return patterns_detected if patterns_detected else ["No significant patterns detected"]

    def calculate_technical_indicators(self, data):
        close = data['Close']
        high = data['High']
        low = data['Low']

        indicators = {}
        try:
            indicators['rsi'] = self.clean_value(talib.RSI(close.values, timeperiod=14)[-1])
            indicators['sma_20'] = self.clean_value(close.rolling(20).mean().iloc[-1])
            indicators['sma_50'] = self.clean_value(close.rolling(50).mean().iloc[-1])
            indicators['ema_20'] = self.clean_value(close.ewm(span=20).mean().iloc[-1])

            macd, signal, hist = talib.MACD(close.values)
            indicators['macd'] = self.clean_value(macd[-1])
            indicators['macd_signal'] = self.clean_value(signal[-1])
            indicators['macd_histogram'] = self.clean_value(hist[-1])

            upper, middle, lower = talib.BBANDS(close.values)
            indicators['bb_upper'] = self.clean_value(upper[-1])
            indicators['bb_middle'] = self.clean_value(middle[-1])
            indicators['bb_lower'] = self.clean_value(lower[-1])

            indicators['atr'] = self.clean_value(talib.ATR(high.values, low.values, close.values, timeperiod=14)[-1])
            indicators['avg_volume'] = self.clean_value(data['Volume'].tail(20).mean())
            indicators['current_volume'] = self.clean_value(data['Volume'].iloc[-1])

        except Exception as e:
            logger.error(f"Technical indicator error: {e}")

        return indicators

    def generate_ai_recommendation(
        self, current_price, indicators, vwap, supertrend, patterns, fib_data, sr_data
    ):
        score = 0
        signals = []
        try:
            if fib_data and fib_data.get('most_important_levels'):
                for fib_level in fib_data['most_important_levels'][:2]:
                    if fib_level['type'] == 'Support' and current_price > fib_level['price']:
                        score += 1
                        signals.append(
                            f"Strong Fib Support at {fib_level['price']} ({fib_level['touches']} touches)"
                        )
                    elif fib_level['type'] == 'Resistance' and current_price < fib_level['price']:
                        score -= 1
                        signals.append(
                            f"Strong Fib Resistance at {fib_level['price']} ({fib_level['touches']} touches)"
                        )

            if sr_data and sr_data.get('most_active_level'):
                active_level = sr_data['most_active_level']
                if active_level['type'] == 'support':
                    score += 1
                    signals.append(
                        f"Strong Support at {active_level['price']} ({active_level['buy_reactions']} buy reactions)"
                    )
                else:
                    score -= 1
                    signals.append(
                        f"Strong Resistance at {active_level['price']} ({active_level['sell_reactions']} sell reactions)"
                    )

            if vwap.get('current_vwap'):
                if current_price > vwap['current_vwap']:
                    score += 1
                    signals.append(f"Price above VWAP ({vwap['current_vwap']}) - Bullish")
                else:
                    score -= 1
                    signals.append(f"Price below VWAP ({vwap['current_vwap']}) - Bearish")

            if supertrend.get('signal') == 'BUY':
                score += 2
                signals.append(f"Supertrend BUY signal at {supertrend.get('supertrend_value')}")
            elif supertrend.get('signal') == 'SELL':
                score -= 2
                signals.append(f"Supertrend SELL signal at {supertrend.get('supertrend_value')}")

            rsi = indicators.get('rsi')
            if rsi:
                if rsi < 30:
                    score += 2
                    signals.append(f"RSI Oversold ({rsi:.1f}) - Strong Buy")
                elif rsi > 70:
                    score -= 2
                    signals.append(f"RSI Overbought ({rsi:.1f}) - Strong Sell")

            if indicators.get('macd') and indicators.get('macd_signal'):
                if indicators['macd'] > indicators['macd_signal']:
                    score += 1
                    signals.append("MACD Bullish crossover")
                else:
                    score -= 1
                    signals.append("MACD Bearish crossover")

            for pattern in patterns:
                if "Bullish" in pattern:
                    score += 0.5
                    signals.append(pattern)
                elif "Bearish" in pattern:
                    score -= 0.5
                    signals.append(pattern)

            if score >= 4:
                recommendation = "STRONG BUY"
                confidence = min(95, 65 + (score * 4))
            elif score >= 2:
                recommendation = "BUY"
                confidence = min(85, 60 + (score * 4))
            elif score <= -4:
                recommendation = "STRONG SELL"
                confidence = min(95, 65 + (abs(score) * 4))
            elif score <= -2:
                recommendation = "SELL"
                confidence = min(85, 60 + (abs(score) * 4))
            else:
                recommendation = "HOLD"
                confidence = 50

        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            recommendation = "HOLD"
            confidence = 50
            signals = ["Error generating signals"]

        return recommendation, confidence, signals

    async def fetch_stock_news(self, stock_name):
        articles = []
        try:
            symbol = self.stock_symbols.get(stock_name.upper())
            if symbol:
                ticker = yf.Ticker(symbol)
                news_data = ticker.news
                for item in news_data[:10]:
                    article = {
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'url': item.get('link', ''),
                        'source': 'Yahoo Finance',
                        'published_date': datetime.fromtimestamp(
                            item.get('providerPublishTime', time.time())
                        ).isoformat()
                    }
                    articles.append(article)
        except Exception as e:
            logger.error(f"News fetch error: {e}")
        return articles


# ================== FASTAPI APP ===================

app = FastAPI(
    title="Advanced Trading Analysis API with Upstox LTP",
    version="6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = AdvancedStockAnalyzer()
upstox_fetcher = UpstoxDataFetcher(UPSTOX_ACCESS_TOKEN)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "6.0",
        "features": [
            "Upstox LTP",
            "Option LTP (placeholder)",
            "Chart Data",
            "1H Fibonacci",
            "1D Support/Resistance"
        ]
    }


@app.get("/stocks")
async def get_stocks():
    return {"success": True, "stocks": list(analyzer.stock_symbols.keys())}


# --------- UPSTOX: LTP & OPTIONS ENDPOINTS ----------

@app.get("/get_ltp")
async def get_ltp(instrument_key: str):
    """
    Simple Upstox LTP endpoint.
    Example: instrument_key = NSE_EQ|INE062A01020
    """
    ltp = upstox_fetcher.get_live_ltp(instrument_key)
    if ltp == 0:
        raise HTTPException(500, "Unable to fetch LTP")
    return {"instrument_key": instrument_key, "ltp": ltp}


@app.get("/get_option_ltp")
async def get_option_ltp(
    underlying_instrument_key: str,
    expiry: str,
    strike: float,
    option_type: str
):
    """
    Options LTP endpoint.
    Abhi ke liye mapping dummy hai, sirf underlying key se LTP de raha hai.
    Baad me:
    - instruments API/CSV se expiry+strike+CE/PE -> instrument_key banao. [web:2]
    """
    ltp = upstox_fetcher.get_option_ltp(
        underlying_instrument_key, expiry, strike, option_type
    )
    if ltp == 0:
        raise HTTPException(500, "Unable to fetch option LTP")
    return {
        "underlying_instrument_key": underlying_instrument_key,
        "expiry": expiry,
        "strike": strike,
        "option_type": option_type.upper(),
        "ltp": ltp
    }


# --------- CHART DATA (yfinance) ----------

@app.get("/chart_data/{stock_name}")
async def get_chart_data(stock_name: str):
    try:
        symbol = analyzer.stock_symbols.get(stock_name.upper())
        if not symbol:
            raise HTTPException(404, f"Stock {stock_name} not found")
        candles = analyzer.get_chart_data(symbol, period='1y', interval='1d')
        if not candles:
            raise HTTPException(500, "Unable to fetch chart data")
        return {
            "success": True,
            "stock_name": stock_name.upper(),
            "symbol": symbol,
            "candles": candles,
            "total_candles": len(candles)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(500, str(e))


# --------- ADVANCED ANALYSIS (yfinance) ----------

@app.get("/analyze/{stock_name}")
async def analyze_stock(stock_name: str):
    try:
        symbol = analyzer.stock_symbols.get(stock_name.upper())
        if not symbol:
            raise HTTPException(404, f"Stock {stock_name} not found")

        data_1h, data_1d, data_long, info = analyzer.get_stock_data_multiple_timeframes(symbol)
        if data_1h is None or data_1d is None:
            raise HTTPException(500, "Unable to fetch stock data")

        current_price = float(data_1d['Close'].iloc[-1])
        prev_close = float(data_1d['Close'].iloc[-2])

        fibonacci_1h = analyzer.calculate_fibonacci_1h_advanced(data_1h)
        support_resistance_1d = analyzer.calculate_support_resistance_1d_advanced(
            data_1d, current_price
        )

        vwap = analyzer.calculate_vwap(data_long.tail(252))
        supertrend = analyzer.calculate_supertrend(data_long.tail(252))
        indicators = analyzer.calculate_technical_indicators(data_long)
        candlestick_patterns = analyzer.detect_candlestick_patterns(data_long.tail(100))

        recommendation, confidence, signals = analyzer.generate_ai_recommendation(
            current_price, indicators, vwap, supertrend,
            candlestick_patterns, fibonacci_1h, support_resistance_1d
        )

        news = await analyzer.fetch_stock_news(stock_name)

        is_index = stock_name.upper() in ['NIFTY', 'BANKNIFTY']
        market_cap = "N/A"
        pe_ratio = "N/A"

        if not is_index and info:
            mc = info.get('marketCap', 0)
            if mc:
                market_cap = f"₹{mc/1e9:.2f}K Cr"
            if info.get('trailingPE'):
                pe_ratio = round(info['trailingPE'], 1)

        atr_value = indicators.get('atr') or (current_price * 0.02)

        return {
            "success": True,
            "stock_name": stock_name.upper(),
            "symbol": symbol,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "realtime_data": {
                "current_price": round(current_price, 2),
                "change": round(current_price - prev_close, 2),
                "change_percent": round(((current_price - prev_close) / prev_close) * 100, 2),
                "volume": int(data_1d['Volume'].iloc[-1]),
                "market_cap": market_cap,
                "pe_ratio": pe_ratio
            },
            "technical_indicators": {
                "rsi": indicators.get('rsi') or 50.0,
                "macd": indicators.get('macd') or 0.0,
                "macd_signal": indicators.get('macd_signal') or 0.0,
                "sma_20": indicators.get('sma_20') or current_price,
                "ema_20": indicators.get('ema_20') or current_price,
                "atr": atr_value
            },
            "vwap": {
                "current_vwap": vwap.get('current_vwap') or current_price,
            },
            "supertrend": {
                "supertrend_value": supertrend.get('supertrend_value') or current_price,
                "signal": supertrend.get('signal', 'HOLD')
            },
            "fibonacci_analysis_1h": fibonacci_1h,
            "support_resistance_analysis_1d": support_resistance_1d,
            "candlestick_patterns": candlestick_patterns,
            "targets_stoploss": {
                "target1": round(current_price + (atr_value * 2), 2),
                "target2": round(current_price + (atr_value * 3), 2),
                "stop_loss": round(current_price - (atr_value * 1.5), 2)
            },
            "ai_recommendation": {
                "action": recommendation,
                "confidence": confidence,
                "signals": signals
            },
            "news": news
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    logger.info("Starting API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


