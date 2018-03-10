from __future__ import division
import pandas as pd 
import numpy as np 
import warnings
from collections import Counter 
import random

def k_nearest_neighbors(data,predict,k):
	if len(data) >= k:
		warnings.warn('K is very low')
	distances=[]
	for grp in data:
		for features in data[grp]:
			eucl_dist=np.linalg.norm(np.array(features)-np.array(predict))#calculates the euclidean distance
			distances.append([eucl_dist,grp])
	votes=[i[1] for i in sorted(distances) [:k]]
	vote=Counter(votes).most_common(1)[0][0] 
	#confidence=Counter(votes).most_common(1)[0][1]/k
	#print(confidence)
	return vote


df=pd.read_csv('dataset1.data')
df.replace('?',-99999,inplace=True)#modify missing values
df.drop(['id'],1,inplace=True)
full_data=df.astype(float).values.tolist()
random.shuffle(full_data)

test_size=0.4
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])
for i in test_data:
	test_set[i[-1]].append(i[:-1])
correct=0
total=0
for group in test_set:
	for data in test_set[group]:
		vote=k_nearest_neighbors(train_set,data,5)
		if vote == group:
			correct=correct+1
		total=total+1
print('Accuracy:',correct/total)