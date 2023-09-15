import random
import numpy as np
import torch

from data_prep import get_data
from svm import *
from utils import *

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # x, _, _ = get_data(use_tfidf=False)
    # print(x.shape)
    # svm(False, False)# unrelated_weight=1.4592)
    # x, _, _ = get_bert_embeddings_cpu()
    # print(x.shape)
    rerun()