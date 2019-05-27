import re
import numpy as np
#
# print(np.random.permutation(np.arange(13576)))
from tqdm import tqdm,trange
from time import sleep
# text = ""
# pbar = tqdm(["a", "b", "c", "d"])
# for char in pbar:
#     pbar.set_description("Processing %s" % char)
import pandas as pd
np.random.seed(1)
# df = pd.DataFrame({'A': [1, 1, 2, 2],
#                   'B': [1, 2, 3, 4],
#                   'C': np.random.randn(4)})
# print(df.head())
x="LdrGetProcedureAddress"
# a=" ".join(list(x))
# print(a)
# c=x.split(" ")
# d=x.count(x)
# print(d/len(x))
# data=pd.DataFrame({"q1":["1","2","3","4","5"],
#                     "q2": ["2", "3", "5", "4", "5"]})
# print(data)
# que=pd.DataFrame({"qid":["1","2","3","4","5"],
#                     "words": ["W17378 W17534 W03249 W01490 W18802", "W17378 W08158 W20171 W11246 W14759", "W11385 W14103 W02556 W13157 W09749", "W17508 W18238 W02952 W18103", "W17508 W18238 W02952 W18103"]})
# print(que)
#
# c=pd.merge(data,que[['qid','words']],left_on="q1",right_on='qid',how='left')
# c=pd.merge(c,que[['qid','words']],left_on='q2',right_on='qid',how='left')
#
# c.drop(['qid_x','qid_y'],axis=1,inplace=True)
#
# print(c.to_csv("./data/pd.csv",index=False))
from sklearn.model_selection import StratifiedKFold
x=np.array([
    [[1,2,3,4,5,3,4,5],
    [11,12,13,14,6,3,4,5]],

    [[21,22,23,24,6,3,4,5],
    [31,32,33,34,6,3,4,5]],

    [[41,42,43,44,6,3,4,5],
    [51,52,53,54,6,3,4,5]],
])
# print(x[[0,1,2]][:,0])
# y=np.array([1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
# #n_folds这个参数没有，引入的包不同，
#
# sfolder = StratifiedKFold(n_splits=5,random_state=0,shuffle=False)
#
# for train, test in sfolder.split(X,y):
#     print('Train: %s | test: %s' % (train, test))
#     print(" ")
#
aa=[
    [1,2,3,4],
    [4,5,6,7]
]
c=np.mean(aa,axis=0)
print(c)