import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
from sklearn.cluster import KMeans  

x=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

clf=KMeans(n_clusters=4)
clf.fit(x)

centroids=clf.cluster_centers_#position of clusters
labels=clf.labels_#gives the number of clusters
colors=10*["g.","r.","b."]
for i in range(len(x)):
	plt.plot(x[i][0],x[i][1],colors[labels[i]],markersize=10)
plt.scatter(centroids[:,0],centroids[:,1],s=150,linewidth=5)
#plt.scatter(x[:,0],x[:,1],s=150,linewidth=5)
plt.show()