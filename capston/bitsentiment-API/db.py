from datetime import datetime, date
from typing import Set
import sqlite3

con = sqlite3.connect('tweets.db')
cur = con.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS tweets (
    ID          INTEGER PRIMARY KEY,
    content     TEXT NOT NULL,
    created_at  TIMESTAMP,
    ref_coins   TEXT NOT NULL
)''')
cur.execute('''CREATE TABLE IF NOT EXISTS scores (
    coin_name   TEXT NOT NULL,
    tweet_count INT,
    score       INT,
    created_at  TIMESTAMP
)''')
cur.execute('''CREATE TABLE IF NOT EXISTS fngs (
    fng         INT,
    created_at  TIMESTAMP
)''')


def commit_db():
    con.commit()


def insert_tweet(tweet_id: int, content: str, created_at: datetime, ref_coins: Set[str]):
    cur.execute('INSERT OR IGNORE INTO tweets VALUES (?,?,?,?)', (tweet_id, content, created_at, ",".join(ref_coins)))


def insert_result(coin_name: str, tweet_count: int, score: float, created_at: datetime):
    cur.execute('INSERT INTO scores VALUES (?,?,?,?)', (coin_name, tweet_count, score, created_at))
    con.commit()


def get_results():
    return cur.execute("SELECT * FROM scores WHERE DATETIME(created_at,'+1 day') > DATE('now')")


def get_fear_and_greed() -> float:
    return cur.execute("SELECT fng FROM fngs ORDER BY created_at DESC LIMIT 1").fetchone()


def set_fear_and_greed(fear_and_greed: float, created_at: datetime):
    cur.execute('INSERT INTO fngs VALUES (?,?)', (fear_and_greed, created_at))
    con.commit()


def get_total_market_cap(when: date):
    data = [
        [date(2022, 3, 17), 1906.61],
        [date(2022, 3, 24), 2045.45],
        [date(2022, 3, 31), 2245.1],
        [date(2022, 4, 7), 2124.71],
        [date(2022, 4, 14), 2010.58],
        [date(2022, 4, 21), 2017.7],
        [date(2022, 4, 28), 1903.46],
        [date(2022, 5, 5), 1896.42],
        [date(2022, 5, 12), 1324.56],
        [date(2022, 5, 17), 1354.54],
    ]
    market_cap = data[0]
    for _date, cap in data:
        if when < _date:
            break
        market_cap = cap
    return market_cap * 1000 * 1000 * 1000
