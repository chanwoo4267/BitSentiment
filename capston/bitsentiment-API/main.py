import asyncio
import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from coins import coin_list, coin_names
from db import get_results, insert_result, insert_tweet, commit_db, set_fear_and_greed, get_fear_and_greed, get_total_market_cap
from sentiment_analysis import get_emotion
from twitter import get_coin_tweets
from ad_classification import is_ad

tweets_per_hour = 1000

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_headers=["*"],
)

fear_and_greed: float
coin_emotion = {
    "BTC": 50,
    "ETH": 50,
    "USDT": 50,
    "USDC": 50,
    "BNB": 50,
    "XRP": 50,
    "ADA": 50,
    "BUSD": 50,
    "SOL": 50,
    "DOGE": 50,
}
coin_bitcoin = "https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
coin_bitcoin_history = "https://api.coingecko.com/api/v3/coins/bitcoin/history?date={}-{}-{}&localization=false"
coin_bitcoin_chart = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=60"
coin_global = "https://api.coingecko.com/api/v3/global"
coin_ticker = [
    "https://api.alternative.me/v2/ticker/bitcoin/",
    "https://api.alternative.me/v2/ticker/ethereum/",
    "https://api.alternative.me/v2/ticker/tether/",
    "https://api.alternative.me/v2/ticker/bitcoin/",
    "https://api.alternative.me/v2/ticker/bitcoin/",
    "https://api.alternative.me/v2/ticker/bitcoin/",
    "https://api.alternative.me/v2/ticker/bitcoin/",
    "https://api.alternative.me/v2/ticker/bitcoin/",
    "https://api.alternative.me/v2/ticker/bitcoin/",
    "https://api.alternative.me/v2/ticker/bitcoin/",
]


def get_score_100(count, score):
    return 50 * (count + score) / count


def set_coin_cache(coin_tweet_count, coin_score):
    for coin_name in coin_names:
        if coin_tweet_count[coin_name] < 1:
            coin_emotion[coin_name] = 0
        else:
            coin_emotion[coin_name] = get_score_100(coin_tweet_count[coin_name], coin_score[coin_name])


def load_coin_data():
    coin_tweet_count = dict.fromkeys(coin_names, 0)
    coin_score = dict.fromkeys(coin_names, 0)
    for coin_name, tweet_count, score, created_at in get_results():
        coin_tweet_count[coin_name] += tweet_count
        coin_score[coin_name] += score
    set_coin_cache(coin_tweet_count, coin_score)


def update_coin_data():
    tweets = get_coin_tweets(tweets_per_hour)
    coin_tweet_count = dict.fromkeys(coin_names, 0)
    coin_score = dict.fromkeys(coin_names, 0)
    for tweet_id, tweet in tweets.items():
        insert_tweet(tweet_id, tweet[0], tweet[1], tweet[2])
        if not is_ad(tweet[0]):
            score = get_emotion(tweet[0])
            for coin_name in tweet[2]:
                coin_tweet_count[coin_name] += 1
                coin_score[coin_name] += score
    times = datetime.now()
    times = times.replace(minute=0, second=0, microsecond=0)
    for coin_name in coin_names:
        insert_result(coin_name, coin_tweet_count[coin_name], coin_score[coin_name], times)
    commit_db()


def load_fear_and_greed():
    global fear_and_greed
    fng = get_fear_and_greed()
    if fng is None:
        fear_and_greed = 50
    else:
        fear_and_greed = fng[0]


def update_fear_and_greed():
    dt_now = datetime.now()
    dt_2m_ago = dt_now - relativedelta(months=2)
    coin_bitcoin_2m_ago = coin_bitcoin_history.format(dt_2m_ago.day, dt_2m_ago.month, dt_2m_ago.year)
    res_bitcoin = requests.get(coin_bitcoin)
    res_bitcoin_2m_ago = requests.get(coin_bitcoin_2m_ago)
    res_bitcoin_chart = requests.get(coin_bitcoin_chart)
    res_global = requests.get(coin_global)
    if not res_bitcoin.ok or not res_bitcoin_2m_ago.ok or not res_bitcoin_chart.ok or not res_global.ok:
        return
    # Market Momentum: 코인을 구매하는 사람이 많을 경우 greed
    volume_sum = 0
    volume_last = 0
    volumes = res_bitcoin_chart.json().get("total_volumes")
    for _, vol in volumes:
        volume_sum += vol
        volume_last = vol
    market_momentum_score = min(100, 50 * volume_last / (volume_sum / len(volumes)))
    # Safe haven: 안정적인 코인의 비율이 늘어나면 fear
    cap_p_bitcoin = res_global.json().get("data").get("market_cap_percentage").get("btc")
    caps = res_bitcoin_chart.json().get("market_caps")
    cap_sum = 0
    for _, cap in caps:
        cap_sum += cap
    cap_avg = cap_sum / len(caps)
    avg_p = cap_avg / get_total_market_cap(date(dt_2m_ago.year, dt_2m_ago.month, dt_2m_ago.day)) * 100
    change_rate = cap_p_bitcoin / avg_p * 100 - 100
    safe_haven_score = max(0, min(100, 50 - change_rate))
    # Volatility: 비트코인 시세 변동이 높을 경우 fear
    bitcoin_diff = res_bitcoin.json().get("market_data").get("price_change_percentage_60d")
    bitcoin_diff_max = -40  # Source: RBC GAM, Bloomberg
    volatility_score = max(0, min(100, 100 * (1 - abs(bitcoin_diff / bitcoin_diff_max))))
    # SNS: 트위터의 코인 관련 트윗을 분석하여 부정적인 언급이 많을 경우 fear
    tweet_counts = 0
    tweet_scores = 0
    for _, count, score, _ in get_results():
        tweet_counts += count
        tweet_scores += score
    sns_tweet_score = get_score_100(tweet_counts, tweet_scores)
    fng = 0.25 * market_momentum_score + 0.25 * safe_haven_score + 0.25 * volatility_score + 0.25 * sns_tweet_score
    times = datetime.now()
    times = times.replace(minute=0, second=0, microsecond=0)
    print(f"[{datetime.now()}] Fear and greed index: {fng}, Market Momentum: {market_momentum_score}, Safe haven: {safe_haven_score}, Volatility: {volatility_score}, SNS: {sns_tweet_score}")
    set_fear_and_greed(fng, times)

async def update_every_hour():
    seconds = 60 * (60 - datetime.now().minute) - datetime.now().second
    print(f"[{datetime.now()}] next update scheduled after {seconds} seconds")
    await asyncio.sleep(seconds)
    while True:
        print(f"[{datetime.now()}] update data start")
        update_coin_data()
        load_coin_data()
        update_fear_and_greed()
        load_fear_and_greed()
        print(f"[{datetime.now()}] update data end")
        seconds = 60 * (60 - datetime.now().minute) - datetime.now().second
        print(f"[{datetime.now()}] next update scheduled after {seconds} seconds")
        await asyncio.sleep(seconds)


@app.get("/api/main")
async def send_fng_emotion():
    return {
        "fng": fear_and_greed,
        "coin": coin_emotion
    }


@app.on_event("startup")
async def create_every_hour_task():
    load_coin_data()
    load_fear_and_greed()
    asyncio.create_task(update_every_hour())
