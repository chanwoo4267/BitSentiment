from typing import Dict, Set, Tuple
import re
import tweepy
from datetime import datetime
from coins import coin_list

# Twitter API private key
consumer_key = "s1l05YA0ujr3EVM26HPtn75GG"
consumer_secret = "Vhy9xrmnVfLuL2BvV3UstPAKbIsZ7stAujeSaj2oWoout8O8iy"
access_token = "1509421128938573830-7yzNs1DEervoHTlkHHmxvWWNYP1xYr"
access_token_secret = "LRbbC8KjLs73xWVsOpP4alHrwmCZHPO1I7UFHqyuArxrx"

# exclude from search
exclude = ["giveaway", "reward", "lottery", "airdrop"]


def _find_ref(tweet: str) -> Set[str]:
    refs = set()
    for coin in coin_list:
        tweet_lower = tweet.lower()
        if coin[0].lower() in tweet_lower or coin[1].lower() in tweet_lower:
            refs.add(coin[0])
    return refs


def _clean_text(data):
    # remove emojis - 🧐
    data = _remove_emojis(data)
    # remove url - https://asdf.com
    data = re.sub(r"http[s]?://(?:[a-zA-Z0-9$-_@.&+!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+…?", ' ', data)
    # Remove Garbage Words (ex. &lt, &gt, etc)
    data = re.sub('&+[a-z]+', ' ', data)
    # Remove multi spacing & Reform sentence
    data = ' '.join(data.split())

    return data


def _remove_emojis(data):
    pattern = re.compile("["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                         u"\U00002500-\U00002BEF"  # chinese char
                         u"\U00002702-\U000027B0"
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         u"\U0001f926-\U0001f937"
                         u"\U00010000-\U0010FFFF"
                         u"\u2640-\u2642"
                         u"\u2600-\u2B55"
                         u"\u200d"
                         u"\u23cf"
                         u"\u23e9"
                         u"\u231a"
                         u"\ufe0f"  # dingbats
                         u"\u3030"
                         "]+", re.UNICODE)
    return re.sub(pattern, '', data)


def get_coin_tweets(number: int) -> Dict[int, Tuple[str, datetime, Set[str]]]:

    # OAuth handler creation and privacy auth request
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    # Access request
    auth.set_access_token(access_token, access_token_secret)

    # api instance creation
    api = tweepy.API(auth, wait_on_rate_limit=True)

    search_result = dict()
    query_str = ""
    for coin in coin_list:
        query_str += f"{coin[0]} OR {coin[1]} OR "
    query_str = query_str.removesuffix("OR ")
    query_str += "-" + " -".join(exclude) + " -filter:retweets"
    for status in tweepy.Cursor(api.search_tweets, q=query_str, lang="en").items(number):
        clean_text = _clean_text(status.text)
        refs = _find_ref(clean_text)
        if len(refs) > 0:
            search_result[status.id] = (clean_text, status.created_at, refs)

    return search_result
