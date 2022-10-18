import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import transformers
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TFTrainer, TFTrainingArguments

pd.set_option('display.max_colwidth', None)
BATCH_SIZE = 16
N_EPOCHS = 3

df = pd.read_csv('twitter.csv', names=["message", "label"], encoding='cp949')

X = list(df['message'])
y = list(df['label'])
y = list(pd.get_dummies(y,drop_first=True)[True])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

#chose the optimizer
optimizerr = tf.keras.optimizers.Adam(learning_rate=5e-5)

#define the loss function 
losss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#build the model
model.compile(optimizer=optimizerr,
              loss=losss,
              metrics=['accuracy'])

history = model.fit(train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE),
          epochs=N_EPOCHS,
          batch_size=BATCH_SIZE)

# model evaluation on the test set
model.evaluate(test_dataset.shuffle(len(X_test)).batch(BATCH_SIZE), 
               return_dict=True, 
               batch_size=BATCH_SIZE)

# tests
def predict_proba(text_list, model, tokenizer):  
    #tokenize the text
    encodings = tokenizer(text_list, 
                          max_length=1000, 
                          truncation=True, 
                          padding=True)
    #transform to tf.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
    #predict
    preds = model.predict(dataset.batch(1)).logits  
    
    #transform to array with probabilities
    res = tf.nn.softmax(preds, axis=1).numpy()      
    
    return res

examples = [
    'In 2008, several failing banks were bailed out partially using taxpayer money. Putting all money at bank provide risk, risk of devaluation, risk of inflation, risk of aggressive centralise policy. Decentralized system like bitcoin working on blockchain provide relief.',
    'Bitcoin is counterfeit. Disagree? Look again.',
    'Did I make a bad GPU purchase before the end of Ethereum mining?',
    '@Mamooetz Help. I created this bot to reply to ETH, BITCOIN, and NFT but I dont know how to shut it off.',
    '@WaldorickWilson Cryptocurrency doesnt have to be cryptic. Luno takes the complexity out of #Bitcoin and lets you buy, store, learn and earn all in one place',
    '@bridgesplit @coinfund_io @jumpcapital @solana @coinbase @ruckerparkvc @packyM @a41_Ventures @Sfermion_ @APompliano The project was executed in a very professional manner and had a clear development plan. Made by a very professional and experienced team. Without a doubt, this is one of the greatest projects @Anwar3Hossain @ShibbirAkash @anwar1hossain',
    '@Stepnofficial @Barndog_Solana After meeting STEPN, I got into the habit of walking every day. Thank you for the wonderful project.',
    '@RichardHeartWin @HalloFeld Is it better to DCA into hex or wait till Bitcoin bottoms to grab a bag? Not financial advice of course but if you were me',
    'Verifying transaction for @gdsaf on @Solana. Trial tokens claim id: 43242152178 $SOL',
    'Limited-time deal: Invicta Mens Best Diver stainless metal Quartz Watch with ... via @amazon',
    '@abraham Hello Bitcoin Pizza Day Hello Bitcoin Pizza Day from #silvanas Join Here @Sgdsfdrr @ampewflputri @paskjmgds_',
]

result = predict_proba(examples, model, tokenizer)
print(result)

# save model
dataset_name = 'adv'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

# model.save(saved_model_path)
tf.saved_model.save(model, saved_model_path)
model.save(model, saved_model_path + 'aa')

# load model
loaded_model = tf.saved_model.load(saved_model_path)
inference_function = loaded_model.signatures['serving_default']

reloaded = tf.keras.models.load_model(saved_model_path)