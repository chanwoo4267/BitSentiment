import pandas as pd
import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pickle

# read saved dataset (twitter.csv)
df = pd.read_csv('twitter.csv', names=["message", "label"], encoding='cp949')
x = list(df['message'])
y = list(df['label'])
y = list(pd.get_dummies(y,drop_first=True)[True])

MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 1000
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def construct_encodings(x, tkzr, max_len, trucation=True, padding=True):
    return tkzr(x, max_length=max_len, truncation=trucation, padding=padding)
    
encodings = construct_encodings(x, tokenizer, max_len=MAX_LEN)

def construct_tfdataset(encodings, y=None):
    if y:
        return tf.data.Dataset.from_tensor_slices((dict(encodings),y))
    else:
        # this case is used when making predictions on unseen samples after training
        return tf.data.Dataset.from_tensor_slices(dict(encodings))
    
tfdataset = construct_tfdataset(encodings, y)

TEST_SPLIT = 0.2
BATCH_SIZE = 2

train_size = int(len(x) * (1-TEST_SPLIT))

#tfdataset = tfdataset.shuffle(len(x))
tfdataset_train = tfdataset.take(train_size)
tfdataset_test = tfdataset.skip(train_size)

tfdataset_train = tfdataset_train.batch(BATCH_SIZE)
tfdataset_test = tfdataset_test.batch(BATCH_SIZE)

# fine-tuning
N_EPOCHS = 10

model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
optimizer = optimizers.Adam(learning_rate=3e-5)
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(tfdataset_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS)

# test model
benchmarks = model.evaluate(tfdataset_test, return_dict=True, batch_size=BATCH_SIZE)
print(benchmarks)

def create_predictor(model, model_name, max_len):
  tkzr = DistilBertTokenizer.from_pretrained(model_name)
  def predict_proba(text):
      x = [text]

      encodings = construct_encodings(x, tkzr, max_len=max_len)
      tfdataset = construct_tfdataset(encodings)
      tfdataset = tfdataset.batch(1)

      preds = model.predict(tfdataset).logits
      preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
      return preds[0][0]
    
  return predict_proba

clf = create_predictor(model, MODEL_NAME, MAX_LEN)

print(clf('Sale Alert @Reveals June 3rd bought for ETH ( USD) by The-Parlour from 0x764f15 @badbearsmsk #beefrens #NFT'))
print(clf('of DOGE RARES are unavailable. Cheapest - ; Series 2, Card #41 "DOGESTAN" for 0.5 DOGE. Most - ; Series 1, Card #5 "COUPDDOGE" for 61776 ($ DOGE. (XDP): 3019968.72898249 - Jun 3, 2022 7:43 AM RAREs For Sale:'))
print(clf('Otherdeed #85470 was purchased for 3.1 ETH'))
print(clf('Check out my new NFT on OpenSea! via @opensea #NFT #NFTCommunity #NFTGiveaway #nftart #PolygonNFT #NFTartist #nftcollectors #NFTs #NFTProject #ArtistOnTwitter #rarible #ethereum #Polygon #flower #tezos #CleanNFT #nonfungible #DigitalAssets #artist'))
print(clf("the hook. That's pretty much why Solana still exists and how the majority of the market thinks and works. Eventually someone gets stuck with an ugly broken chair worth way too much. Maybe I am weird but I don't want that to happen to me so... 2/3"))
print(clf('@IIICapital The bleeding that every alt will see in the year or so will be Glorious to watch. #Bitcoin is sucking the energy out of them each day!'))
print(clf('How #Bitcoin mining devastated this New York town Between rising electricity rates and soaring climate costs, cryptomining is taking its toll on communities.'))
print(clf('Bitcoin and Capston design is hard to proceed #espga #capstone'))

# save model
model.save_pretrained('./adv_model/clf')
with open('./adv_model/info.pkl', 'wb') as f:
    pickle.dump((MODEL_NAME, MAX_LEN), f)

# load model
new_model = TFDistilBertForSequenceClassification.from_pretrained('./adv_model/clf')
model_name, max_len = pickle.load(open('./adv_model/info.pkl', 'rb'))

clf = create_predictor(new_model, model_name, max_len)
print(clf('Sale Alert @Reveals June 3rd bought for ETH ( USD) by The-Parlour from 0x764f15 @badbearsmsk #beefrens #NFT'))
print(clf('of DOGE RARES are unavailable. Cheapest - ; Series 2, Card #41 "DOGESTAN" for 0.5 DOGE. Most - ; Series 1, Card #5 "COUPDDOGE" for 61776 ($ DOGE. (XDP): 3019968.72898249 - Jun 3, 2022 7:43 AM RAREs For Sale:'))
print(clf('Otherdeed #85470 was purchased for 3.1 ETH'))
print(clf('Check out my new NFT on OpenSea! via @opensea #NFT #NFTCommunity #NFTGiveaway #nftart #PolygonNFT #NFTartist #nftcollectors #NFTs #NFTProject #ArtistOnTwitter #rarible #ethereum #Polygon #flower #tezos #CleanNFT #nonfungible #DigitalAssets #artist'))
print(clf("the hook. That's pretty much why Solana still exists and how the majority of the market thinks and works. Eventually someone gets stuck with an ugly broken chair worth way too much. Maybe I am weird but I don't want that to happen to me so... 2/3"))
print(clf('@IIICapital The bleeding that every alt will see in the year or so will be Glorious to watch. #Bitcoin is sucking the energy out of them each day!'))
print(clf('How #Bitcoin mining devastated this New York town Between rising electricity rates and soaring climate costs, cryptomining is taking its toll on communities.'))
print(clf('Fucking Bitcoin and Fucking Capston design #espga #capstone'))