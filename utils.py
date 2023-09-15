import numpy as np
import torch
import os
import re

from root import rootpath

def get_tfidf_embeddings(use_artificial_data=True):
    path = f"{rootpath}./embeddings/embeddings_artificial_tfidf.npz" if \
        use_artificial_data else f"{rootpath}./embeddings/embeddings_tfidf.npz"
    f = np.load(path)
    filenames = f.files
    x = f[filenames[0]]
    y_related = f[filenames[1]]
    y_stance = f[filenames[2]]
    return x, y_related, y_stance

def rerun():
    new_path = f"{rootpath}./bert_embeddings2/"
    os.makedirs(new_path, exist_ok=True)
    files = os.listdir(f"{rootpath}./embeddings_bert/")
    for f in files:
        num = int(re.findall(r"\d+", f)[0])
        if num <= 49000:
            temp = torch.load(f"{rootpath}./embeddings_bert/{f}").detach()
            for i in range(0, 1000):
                curr = temp[i, :, :]
                torch.save(torch.clone(curr), f"{new_path}./bert_x_{num -1000 + i + 1}.pt")

def get_bert_embeddings(use_artificial_data=True):
    files = os.listdir(f"{rootpath}./embeddings_bert/")
    lst = []
    for f in files:
        num = int(re.findall(r"\d+", f)[0])
        if num < 49000 or (num > 49000  and use_artificial_data):
            temp = torch.load(f"{rootpath}./embeddings_bert/{f}",)
            temp1 = torch.sum(temp[:, :24, :], dim=1)# [:, None, :]
            temp2 = torch.sum(temp[:, 24:, :], dim=1)# [:, None, :]
            x = torch.concat([temp1,temp2], dim=1)
            lst.append(temp)            
    x = torch.concat(lst)
    _, y_related, y_stance = get_tfidf_embeddings(use_artificial_data=True)
    return x, y_related, y_stance

def get_bert_embeddings_cpu(use_artificial_data=True):
    files = os.listdir(f"{rootpath}./embeddings_bert/")
    lst = []
    for f in files:
        num = int(re.findall(r"\d+", f)[0])
        if num < 49000 or (num > 49000  and use_artificial_data):
            temp = torch.load(f"{rootpath}./embeddings_bert/{f}",  map_location=torch.device('cpu'))
            print(temp.dtype)
            exit()
            temp1 = torch.sum(temp[:, :24, :], dim=1)# [:, None, :]
            temp2 = torch.sum(temp[:, 24:, :], dim=1)# [:, None, :]
            x = torch.concat([temp1,temp2], dim=1)
            lst.append(temp)            
    x = torch.concat(lst)
    _, y_related, y_stance = get_tfidf_embeddings(use_artificial_data=True)
    return x, y_related, y_stance

if __name__ == "__main__":
    # x = get_bert_embeddings_cpu()
    # print(x.shape)
    rerun()