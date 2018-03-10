import pandas as pd 
import numpy as np 
import warnings
from matplotlib import style
from collections import Counter
import matplotlib.pyplot as plt 

style.use('fivethirtyeight')
distances=[]
def k_nearest_neighbors(data,predict,k):
	if len(data) >= k:
		warnings.warn('K is very low')
	for grp in data:
		for features in data[grp]:
			eucl_dist=np.linalg.norm(np.array(features)-np.array(predict))#calculates the euclidean distance
			distances.append([eucl_dist,grp])
	votes=[i[1] for i in sorted(distances) [:k]]
	#print(votes)
	vote=Counter(votes).most_common(1)[0][0] #highly useful to count use it whenever necessary
	print(vote)

dataset={'k':[[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}

new_features=[4,5]
k_nearest_neighbors(dataset,new_features,3)
#for i in dataset:
#	for ii in dataset[i]:
#		plt.scatter(ii[0],ii[1],s=100,color=i)
#plt.scatter(new_features[0],new_features[1])
#plt.show()

