"""
Sources:
https://towardsdatascience.com/how-i-handled-imbalanced-text-data-ba9b757ab1d8
https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#why-bert-embeddings

"""

import pandas as pd
import numpy as np
import torch
import random
import re
# import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from textblob.translate import NotTranslated
from sklearn.decomposition import TruncatedSVD
# from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from root import rootpath

SR = random.SystemRandom()
LANGUAGES = ["es", "de", "fr", "ar", "te", "hi", "ja", "fa", "sq", "bg", "nl", "gu", "ig", "kk", "mt", "ps"]

STOP = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

TRAIN_STANCES_CSV = f"{rootpath}./fnc-1/train_stances.csv"
TRAIN_BODIES_CSV = f"{rootpath}./fnc-1/train_bodies.csv"
TEST_STANCES_CSV = f"{rootpath}./fnc-1/competition_test_stances.csv"
TEST_BODIES_CSV = f"{rootpath}./fnc-1/competition_test_bodies.csv"

RELATED_DICT = {"agree":1, "disagree":1, "discuss":1, "unrelated":0}
STANCE_DICT = {"agree":0, "disagree":1, "discuss":2, "unrelated":3}

TFIDF_TOKENIZER = TfidfVectorizer(strip_accents="unicode", decode_error="ignore", lowercase=True, min_df=0.01)
SVD = TruncatedSVD(n_components=300, algorithm="arpack", random_state=42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BERT = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
# TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
BERT = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
BERT.eval()
BERT.to(DEVICE)

def get_tfidf_embedding(x, fit):
    x = TFIDF_TOKENIZER.fit_transform(x) if fit else TFIDF_TOKENIZER.transform(x)
    x = SVD.fit_transform(x) if fit else SVD.transform(x)
    return x

def clip_pad_tokens(t, max_size):
    t = t[:max_size]
    num_pad = max_size - len(t)
    if num_pad > 0:
        t += ["[PAD]"] * num_pad
    return t, num_pad

def tokenize_text(head, body):
    head = "[CLS] " + head
    tokens1 = TOKENIZER.tokenize(head)
    tokens1, num_pad = clip_pad_tokens(tokens1, 23)
    tokens1 += ["[SEP]"]
    attention_mask1 = [1] * (len(tokens1) - num_pad - 1) + [0] * num_pad + [1]
    tokens2 = TOKENIZER.tokenize(body)
    tokens2, num_pad = clip_pad_tokens(tokens2, 488)
    attention_mask2 = [1] * (len(tokens2) - num_pad) + [0] * num_pad
    input_ids = TOKENIZER.convert_tokens_to_ids(tokens1 + tokens2)
    token_type_ids = [0] * len(tokens1) + [1] * len(tokens2)
    attention_mask = attention_mask1 + attention_mask2
    # return torch.tensor(input_ids)[None, :], \
    #     torch.tensor(token_type_ids)[None, :], \
    #     torch.tensor(attention_mask)[None, :]
    return torch.tensor(input_ids, device=DEVICE)[None, :], \
        torch.tensor(token_type_ids, device=DEVICE)[None, :], \
        torch.tensor(attention_mask, device=DEVICE)[None, :]

def get_bert_embedding(x, path="embeddings_bert"):
    len_x = len(x) // 2
    embedding_lst = []
    for i in range(len_x):
        input_ids, token_type_ids, attention_mask = tokenize_text(x[i], x[i + len_x])
        with torch.no_grad():
            output_layers = BERT(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[2]
        # embedding = torch.cat([output_layers[-4], output_layers[-3], output_layers[-2], output_layers[-1]], dim=1).squeeze()
        embedding = (output_layers[-4] + output_layers[-3] + output_layers[-2] + output_layers[-1]).squeeze()
        embedding_lst.append(embedding)
        # embedding_lst.append(torch.mean(embedding, dim=0))
        if i > 0 and (i % 1000 == 0 or i == len_x - 1):
            print("Getting it done, bossman.")
            torch.save(torch.stack(embedding_lst), f"{rootpath}/embeddings/{path}_{i}.pt")
            del embedding_lst
            embedding_lst = []
    return torch.stack(embedding_lst)

def clean_text(t):
    t = re.sub(r"[^\w\s*]", " ", t.lower())
    lst = [word for word in t.split() if word not in STOP]
    lst = [LEM.lemmatize(word) for word in lst]
    t = " ".join(lst)
    return t

def clean_data(data):
    data["headline_clean"] = data["headline"].apply(lambda x: clean_text(x))
    data["body_clean"] = data["body"].apply(lambda x: clean_text(x))
    return data

def get_data(use_tfidf, train_data=True):
    data = load_data(train_data=train_data)
    if use_tfidf:
        data = clean_data(data)
        x = pd.concat([data["headline_clean"], data["body_clean"]]).to_numpy().squeeze().astype("U")
        x = get_tfidf_embedding(x, train_data)
        x = np.concatenate([x[:x.shape[0]//2, :], x[x.shape[0]//2:, :]], axis=1)
    else:
        x = pd.concat([data["headline"], data["body"]]).to_numpy().squeeze().astype("U")
        x = get_bert_embedding(x)
    return x, data["related"].to_numpy(), data["stance"].to_numpy()

def load_data(train_data=True, use_artificial_data=True):
    stances = pd.read_csv(TRAIN_STANCES_CSV) if train_data else pd.read_csv(TEST_STANCES_CSV)
    bodies = pd.read_csv(TRAIN_BODIES_CSV) if train_data else pd.read_csv(TEST_BODIES_CSV)
    data = pd.merge(stances, bodies, how="outer", on="Body ID").drop("Body ID", axis=1)
    data.rename(columns={"articleBody":"body", "Headline":"headline", "Stance":"stance"}, inplace=True)
    y = data.drop(["body", "headline"], axis=1).to_numpy().squeeze().astype("U")
    y_related = np.vectorize(RELATED_DICT.get)(y)
    y_stance =  np.vectorize(STANCE_DICT.get)(y)
    data = data.drop(["stance"], axis=1)
    data["related"] = y_related
    data["stance"] = y_stance
    if train_data and use_artificial_data:
        artificial_data = pd.read_csv(f"{rootpath}./data/artificial_data.csv", index_col=0)
        data = pd.concat([data, artificial_data])
    return data

def random_sample(data):
    return list(data.iloc[np.random.randint(data.shape[0])])

def translate_text(text):
    blob = TextBlob(text)
    translated = True
    try:
        lang = SR.choice(LANGUAGES)
        blob = blob.translate(from_lang="en", to=lang)
        blob = blob.translate(from_lang=lang, to="en")
    except NotTranslated:
        translated = False
    return blob.string, translated

def augment_data(data, num):
    artificial_data = []
    for _ in range(num):
        x = random_sample(data)
        headline_translated, translated = translate_text(x[0])
        if translated:
            body_translated, translated = translate_text(x[1])
        if translated:
            stance = x[3]
            artificial_data.append((headline_translated, body_translated, 1, stance))
    return artificial_data

if __name__ == "__main__":
    for _ in range(100):
        data = load_data(use_artificial_data=False)
        data = data[data["related"] == 1]
        artificial_data = augment_data(data, 100)
        df = pd.read_csv(f"{rootpath}./data/artificial_data.csv", index_col=0)
        new_df = pd.DataFrame(artificial_data, columns=["headline", "body", "related", "stance"])
        df = df.append(new_df)
        df.to_csv(f"{rootpath}./data/artificial_data.csv")
