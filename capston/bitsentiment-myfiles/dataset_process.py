import tweepy
import pandas as pd
import re
from csv import writer

def Cleantext(data):
    # remove url - https://asdf.com
    data = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ', data) # http로 시작되는 url
    data = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", ' ', data) # http로 시작되지 않는 url

    # remove mentions - @someone
    #data = re.sub('@[\w_]+', '', data)

    # remove hashtags - #something
    #data = re.sub('#[\w_]+', '', data)

    # remove emojis - 🧐
    data = remove_emojis(data)

    # remove newline
    data = re.sub('\n', ' ', data)

    # Remove Garbage Words (ex. &lt, &gt, etc)
    data = re.sub('[&]+[a-z]+', ' ', data)

    # Remove Special Characters
    # data = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', ' ', data)

    # Remove multi spacing & Reform sentence
    data = ' '.join(data.split())
    return data

def remove_emojis(data):
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

def is_ads(data):
    temp = data.lower()
    if "reward" in temp:
        return True
    if "lottery" in temp:
        return True
    if "giveaway" in temp:
        return True
    if "drops" in temp:
        return True
    else:
        return False

def save_dataframe(list):
    df = pd.DataFrame(list, columns=['Text', 'Label'])
    # dataframe print
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # save data as .csv
    df.to_csv("twitter.csv", index = False)
    print(df)

def append_dataframe(list):
    df = pd.DataFrame(list, columns=['Text', 'Label'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # append data to .csv
    with open("twitter.csv", 'a', newline='', encoding='UTF-8') as f:
        df.to_csv(f, header=(f.tell()==0), index = False)
    print(df)

# check whether text is garbage that is not suitable for studying or judging by BERT
def check_garbage(text):
    text = text.lower()
    if "nft value alert" in text:
        return 0
    if "minted for sol at" in text:
        return 0
    if "was bought for" in text:
        return 0
    if "has been sold" in text:
        return 0
    if "test network" in text:
        return 0
    if "in a user vault at" in text:
        return 0
    if "bitcoinmagazine" in text:
        return 0
    if "opensea" in text:
        return 0

    # too short
    data = re.sub('@[\w_]+', '', text)
    data = re.sub('#[\w_]+', '', data)
    data = ' '.join(data.split())
    if len(data) < 20:
        return 0

    return 1
    

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

    # serach_keyword list
    coin_list = ["bitcoin", "ethereum", "tether", "binance", "USD", "XRP", "solana", "cardano", "doge", "BNB"]

    # tweets list
    tweet_lists = []

    # search keywords in a string & dont search retweets
    coin_list_string = " OR ".join(coin_list) + " -filter:retweets"

    # user input
    item_num = input("input number of tweets : ")
    mode = input("input [create_new] : create new csv file / [append] : append data to existing csv file : ")
    item_num = int(item_num)

    # save information of tweets as [text, label]
    # change parameter q="~" to change search keywords, lang="~" to change search language, items(~) to change search amount

    results = [status._json for status in tweepy.Cursor(api.search_tweets, q=coin_list_string, tweet_mode='extended', lang='en').items(item_num)]
    for result in results:
        text = result["full_text"]
        # cleansing text
        refined_text = Cleantext(text)

        # filter garbage tweets
        if (check_garbage(refined_text) != 0):
        # simple labeling for ads
            label = is_ads(refined_text)
            
            temp_list = [refined_text, label]
            tweet_lists.append(temp_list)

    if mode == "create_new":
        # overwrite existing file
        print("overwrite")
        save_dataframe(tweet_lists)
    elif mode == "append":
        # append data to existing file
        print("append")
        append_dataframe(tweet_lists)
    else:
        print("invalid function")
        
if __name__ == "__main__":
    main()