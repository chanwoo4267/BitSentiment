import tweepy
import pandas as pd
import re
from csv import writer

def main():
    # twitter API private key
    consumer_key = "s1l05YA0ujr3EVM26HPtn75GG"
    consumer_secret = "Vhy9xrmnVfLuL2BvV3UstPAKbIsZ7stAujeSaj2oWoout8O8iy"
    access_token = "1509421128938573830-7yzNs1DEervoHTlkHHmxvWWNYP1xYr"
    access_token_secret = "LRbbC8KjLs73xWVsOpP4alHrwmCZHPO1I7UFHqyuArxrx"

    # OAuth handler creation and privacy auth request
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    # Access request
    auth.set_access_token(access_token, access_token_secret)

    # api instance creation
    api = tweepy.API(auth, wait_on_rate_limit = True)

    coin_list_string = "bitcoin OR btc -filter:retweets"
    tweet_lists = []

    # save information of tweets as [text, created_date, retweets, likes, user_name]
    for status in tweepy.Cursor(api.search_tweets, q=coin_list_string, lang="en").items(100):
        if "btc" not in status.text and "bitcoin" not in status.text:
            continue
        temp_list = [status.id, status.text, status.created_at, status.retweet_count, status.favorite_count, status.user.name]
        tweet_lists.append(temp_list)

    # dataframe print
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df = pd.DataFrame(tweet_lists, columns=['Id', 'Text', 'Created_Date', 'Retweets', 'Likes', 'User'])
    print(df)
    df.to_csv("test_data.csv", index = False)

if __name__ == "__main__":
    main()