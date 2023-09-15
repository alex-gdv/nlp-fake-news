from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import torch

from utils import get_tfidf_embeddings, get_bert_embeddings_cpu
from data_prep import get_data
from root import rootpath

def gridsearch():
    x, y, _, _= get_data(use_tfidf=True, train_data=True)
    parameters = { 
        "C": [0.1, 1.0, 10],
        "gamma": [1, "auto", "scale"],
        "kernel": ["rbf", "linear"]
        }
    model = GridSearchCV(SVC(), parameters, cv=5, n_jobs=-1).fit(x, y)
    print(model.get_params())


def svm(use_tfidf, art):
    # x, y, _ = get_data(use_tfidf=use_tfidf, train_data=True)
    x, y, _ = get_tfidf_embeddings() if use_tfidf else get_bert_embeddings_cpu(use_artificial_data=art)
    print(x.shape)
    unrelated_weight= 2.7 if not art else 1.4592
    svc = SVC(class_weight={1:unrelated_weight})
    svc.fit(x, y)
    suffix = "art" if art else ""
    filename = f"{rootpath}./svc_tfidf_artificial_data.sav" if use_tfidf else f"{rootpath}./svc_bert{suffix}.sav"
    # f"{rootpath}./svc_bert_artificial_data.sav"
    pickle.dump(svc, open(filename, "wb"))