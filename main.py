import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import aiohttp

# إعداد الخادم
app = FastAPI(title="MERIDIAN Quant Engine v4.2")

# تفعيل CORS عشان الـ HTML يقدر يتصل
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# الأصول المالية
ASSETS_YF = {
    "GOLD": "GC=F",       # عقود الذهب
    "SPX": "^GSPC",       # S&P 500
    "DXY": "DX-Y.NYB",    # مؤشر الدولار
}

ASSETS_CRYPTO = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}

# ========== محرك التحليل الرياضي ==========
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
    return round(float(rsi), 1)

def calculate_sma(prices, window=20):
    if len(prices) < window:
        return prices[-1]
    return round(float(np.mean(prices[-window:])), 2)

def calculate_macd(prices):
    if len(prices) < 35:
        return "neutral"
    ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        return "buy"
    elif macd_line.iloc[-1] < signal_line.iloc[-1]:
        return "sell"
    return "neutral"

def calculate_bollinger(prices, window=20):
    if len(prices) < window:
        return "Normal"
    sma = np.mean(prices[-window:])
    std = np.std(prices[-window:])
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    current = prices[-1]
    if current > upper:
        return "Expanding Upper"
    elif current < lower:
        return "Expanding Lower"
    elif std < np.std(prices[-window-5:-5]) * 0.8:
        return "Squeezing"
    return "Normal"

def calculate_market_score(data):
    score = 50
    rsi = data.get("rsi", 50)
    macd = data.get("macd", "neutral")
    whale = data.get("whale", "neutral")

    if rsi > 65: score += 15
    if rsi < 35: score -= 15
    if macd == "buy": score += 15
    if macd == "sell": score -= 15
    if whale == "buy": score += 20
    if whale == "sell": score -= 20

    return max(0, min(100, score))

# ========== جلب البيانات ==========
async def fetch_crypto_prices():
    """جلب أسعار العملات الرقمية من Binance (لحظي)"""
    result = {}
    try:
        async with aiohttp.ClientSession() as session:
            for key, symbol in ASSETS_CRYPTO.items():
                url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = float(data.get("lastPrice", 0))
                        change = float(data.get("priceChangePercent", 0))
                        result[key] = {"price": price, "change": change}
    except Exception as e:
        logging.error(f"Crypto fetch error: {e}")
    return result

def fetch_yf_data():
    """جلب بيانات الأسهم والذهب (متأخر 15 دقيقة لكن مجاني)"""
    result = {}
    for key, ticker in ASSETS_YF.items():
        try:
            asset = yf.Ticker(ticker)
            df = asset.history(period="30d", interval="1d")
            if df.empty or len(df) < 2:
                continue
            close = df["Close"].tolist()
            current = float(close[-1])
            prev = float(close[-2])
            change = round(((current - prev) / prev) * 100, 2)

            result[key] = {
                "price": round(current, 2),
                "change": change,
                "history": [round(p, 2) for p in close[-7:]],
                "rsi": calculate_rsi(close),
                "sma20": calculate_sma(close),
                "macd": calculate_macd(close),
                "bb": calculate_bollinger(close),
            }
        except Exception as e:
            logging.error(f"YF fetch error for {key}: {e}")
    return result

async def build_snapshot():
    """بناء مصفوفة البيانات الكاملة"""
    snapshot = {}

    # جلب العملات الرقمية (لحظية)
    crypto = await fetch_crypto_prices()

    # جلب الأسهم والذهب
    yf_data = fetch_yf_data()
    snapshot.update(yf_data)

    # معالجة العملات الرقمية + إضافة مؤشرات فنية
    for key, data in crypto.items():
        try:
            # جلب بيانات تاريخية من Binance للمؤشرات
            async with aiohttp.ClientSession() as session:
                symbol = ASSETS_CRYPTO[key]
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=30"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
                        closes = [float(k[4]) for k in klines]
                        snapshot[key] = {
                            "price": data["price"],
                            "change": data["change"],
                            "history": [round(p, 2) for p in closes[-7:]],
                            "rsi": calculate_rsi(closes),
                            "sma20": calculate_sma(closes),
                            "macd": calculate_macd(closes),
                            "bb": calculate_bollinger(closes),
                        }
                    else:
                        snapshot[key] = {
                            "price": data["price"],
                            "change": data["change"],
                            "history": [data["price"]] * 7,
                            "rsi": 50, "sma20": data["price"],
                            "macd": "neutral", "bb": "Normal"
                        }
        except Exception as e:
            logging.error(f"Crypto history error for {key}: {e}")
            snapshot[key] = {
                "price": data["price"],
                "change": data["change"],
                "history": [data["price"]] * 7,
                "rsi": 50, "sma20": data["price"],
                "macd": "neutral", "bb": "Normal"
            }

    # إضافة Whale Flow (محاكاة ذكية بناءً على RSI + MACD)
    for key in snapshot:
        rsi = snapshot[key].get("rsi", 50)
        macd = snapshot[key].get("macd", "neutral")
        if rsi > 60 and macd == "buy":
            snapshot[key]["whale"] = "buy"
        elif rsi < 40 and macd == "sell":
            snapshot[key]["whale"] = "sell"
        else:
            snapshot[key]["whale"] = "neutral"

        snapshot[key]["bias"] = macd
        snapshot[key]["score"] = calculate_market_score(snapshot[key])

        # توقعات بسيطة (3 خطوات قادمة)
        current = snapshot[key]["price"]
        trend = 1 if macd == "buy" else (-1 if macd == "sell" else 0)
        vol = current * 0.002 * (1 if key in ["BTC", "ETH"] else 0.0005)
        snapshot[key]["pred"] = [
            round(current + (trend * vol * 1), 2),
            round(current + (trend * vol * 2.5), 2),
            round(current + (trend * vol * 4), 2)
        ]

    return snapshot

# ========== WebSocket Endpoint ==========
@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ MERIDIAN Client Connected")
    try:
        while True:
            data = await build_snapshot()
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(3)  # تحديث كل 3 ثواني
    except WebSocketDisconnect:
        print("❌ Client Disconnected")
    except Exception as e:
        print(f"Error: {e}")

# ========== Health Check ==========
@app.get("/")
def root():
    return {"status": "MERIDIAN Engine v4.2 Running", "time": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
