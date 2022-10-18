import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pickle

tf.get_logger().setLevel('ERROR')

new_model = TFDistilBertForSequenceClassification.from_pretrained('./adv_model/clf')
model_name, max_len = pickle.load(open('./adv_model/info.pkl', 'rb'))


def _construct_encodings(x, tkzr, max_len, trucation=True, padding=True):
    return tkzr(x, max_length=max_len, truncation=trucation, padding=padding)


def _construct_tfdataset(encodings, y=None):
    if y:
        return tf.data.Dataset.from_tensor_slices((dict(encodings), y))
    else:
        # this case is used when making predictions on unseen samples after training
        return tf.data.Dataset.from_tensor_slices(dict(encodings))


def _create_predictor(model, model_name, max_len):
    tkzr = DistilBertTokenizer.from_pretrained(model_name)

    def predict_proba(text):
        x = [text]

        encodings = _construct_encodings(x, tkzr, max_len=max_len)
        tfdataset = _construct_tfdataset(encodings)
        tfdataset = tfdataset.batch(1)

        preds = model.predict(tfdataset).logits
        preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
        return preds[0][0]

    return predict_proba


_clf = _create_predictor(new_model, model_name, max_len)


def is_ad(text: str) -> bool:
    value = _clf(text)
    print(value)
    return value < 0.5
