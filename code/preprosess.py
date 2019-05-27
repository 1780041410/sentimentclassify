import re
import numpy as np
import pandas as pd
from code.langconv import *
def read_trainData(train_path_pos, train_path_neg):
    with open(train_path_pos) as fr_pos, open(train_path_neg) as fr_neg:
        pos_lines = fr_pos.readlines()
        neg_lines = fr_neg.readlines()
        trainData = []
        pos = []
        neg = []
        for line in pos_lines:
            line = simple2tradition(line)
            line=clrAndCutSentence(line)
            pos.append(line.strip("\n"))
        pos = list(set(pos))
        label = ["pos"] * len(pos)
        trainData.extend(pos)
        for line in neg_lines:
            line = simple2tradition(line)
            line = clrAndCutSentence(line)
            neg.append(line.strip("\n"))
        neg = list(set(neg))
        label.extend(["neg"] * len(neg))
        trainData.extend(neg)
    return trainData, label


def simple2tradition(line):
    line = Converter('zh-hans').convert(line)
    return line


def clrAndCutSentence(str):
    sentence = re.sub("[\s+\.\\\`\/_,$%^*(+\"\')]+|[+——()?【】“”！~@#￥%……&*（）]+", "", str)
    return sentence
def dataTotxt(trainData,labels,filename):
    data = pd.DataFrame(columns=["label","content"])
    data["label"]=labels
    data["content"]=trainData
    data.to_csv(filename, index=False,header=None,sep="\t")


if __name__ == "__main__":
    train_path_pos = "../data/pos.csv"
    train_path_neg = "../data/neg.csv"
    train_txt="../data/train.txt"
    test_txt="../data/test.txt"
    trainData, labels = read_trainData(train_path_pos, train_path_neg)
    dataTotxt(trainData, labels,train_txt)
    trainData=trainData[:200]
    labels=labels[:200]
    indices = np.random.permutation(np.arange(len(trainData)))
    testData=np.asarray(trainData)[indices]
    dataTotxt(testData, labels, test_txt)