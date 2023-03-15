# -*- coding: utf-8 -*-

# -- Sheet --

# ## IN this notebook
# 
# **we will be implementing two diff clustering algorithms (KMean & DBSCAN) from scratch upon various datasets then we will provide a short comparison between the two.**
# ## Table of Content📝
# ### Classes are:
#         1-  KMean class
#                 - fit function 
#                 - update centroid
#                 - assaign cluster
#         2-  DBSCAN class
#                 - fit
#                 - get neighbors 
#         3-  From sklearn
#                 - GMM
#                 - Hirarical 
#         
# ### Functions are:
#         1- elbow function
#         2- nearest neighbors function
#         3- ploting function
#         4- decideing GMM Componat
# 
# ### Validation measures:
#         1- F-measure
#         2- Rand statics
#         3- Normlized matual information        


# ### Import important libraris


import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import ward, linkage

# ## class KMeans


class Kmeans:
    def __init__(self,k,max_iter):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
#loop until max_iter or converge        
    def fit(self,data):
    #Step 1 pick random k centroids
        random_init = np.random.choice(len(data),self.k, replace=False)
        self.centroids = data[random_init]
        for i in range(self.max_iter):
            # classfy points into clusters
            cluster_points = self.assign_clusters(data)
            init_centroids = self.centroids #save the random init to compare after
            # update the initi centroids
            self.centroids = self.update_centroids(data,cluster_points)
            # if there is no change in updating centroid break because it mean we converge
            if (init_centroids == self.centroids).all():
                break
        return cluster_points
# calaulate the distance matrix
    #def e_distance(instance , centroid):
     #   return np.sqrt(np.dot(instance-centroid,instance-centroid))

#step 2 assign points to nearest centroids    
    def assign_clusters(self,data):
        cluster_points = []
        distances = []
        for i in data: #loop all row (instance)
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(i-centroid,i-centroid)))
            min_distance = min(distances)
            #here the  distances [] will have k values 
            index_position = distances.index(min_distance) #returns the index of the min_distance.
            #index_position determine wich centroid inctance i belong to
            cluster_points.append(index_position)
            distances.clear()
        return np.array(cluster_points)
#step 3 update centroids
    def update_centroids(self,data,cluster_points):
        new_centroids = []
        cluster_type = np.unique(cluster_points) # store the cluster label
        for type in cluster_type:
            new_centroids.append(data[cluster_points == type].mean(axis=0))# new_centroids store the mean of each cluster_points
        return np.array(new_centroids)   

# ## Function to Find K value using elbow method


from sklearn.cluster import KMeans 
def findoptimk(X):
    # create list to hold SSE values for each k
    SSE = []
    for i in range(2,11):
      # calling and fitting 
      km = KMeans(n_clusters=i,init='k-means++',random_state=42)
      km.fit(X)
      SSE.append(km.inertia_)
    # plot of SSE distances for k  range(n_cluster)
    plt.figure(figsize=(10,8))
    plt.plot(range(2,11),SSE,marker='o',linestyle='--')
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title("ELBOW ")
    plt.show()

# ## Class DBSCAN


# ssed point>> Through it we can create and grow new cluster
class dbscan:
    def __init__(self,eps,MinPts):
        self.eps = eps
        self.MinPts = MinPts
        self.centroids = None       
    def fit(self,data): 
        # set all labels are 0 >> point hasn't been considered yet
        labels = [0]*len(X)
        cluster = 0
        # loop responsible for capturing new seed points 
        for  point in range(len(X)):
            # condition for the label is not 0, continue to the next point 
            if (labels[ point] != 0):
                continue
            # get neighrbos for points 
            neighborPts =  dbscan.getNeighbours(X,  point, self.eps)
            # noise points>> not seed point
            if len(neighborPts) < self.MinPts:
                 labels[ point] = -1
            # this point used in new cluster(seed point) and assign the label to the seed point
            else:
                cluster += 1
                labels[ point] =  cluster
                i = 0
                # neighborPts used as a FIFO queue of points to search, 
                # Looking at each neighbors of Point and search for 
                # new branch points for the cluster. so, will expand the cluster 
                while i < len(neighborPts):
                    # get next point from neighbors
                    newPoint = neighborPts[i]
                    # If newPoint was labelled NOISE, then it's not a branch point (doesn't have enough neighbors)
                    # so make it a leaf(border) point of cluster
                    if labels[newPoint] == -1:
                        labels[newPoint] =  cluster
                    # add new_point to cluster
                    elif labels[newPoint] == 0:
                        labels[newPoint] =  cluster
                        # calling the nweNeighborPts, to find all the neighbors of newPoint 
                        nweNeighborPts =  dbscan.getNeighbours(X, newPoint, self.eps)
                        # add all of its neighbors to queue , it's a branch(core) point! 
                        if len(nweNeighborPts) >= self.MinPts:
                            neighborPts = neighborPts + nweNeighborPts
                    # next point in queue
                    i += 1 
        return labels
    
    # calculate the distance between point and every other 
    # point in the dataset, then returns points within a radius(eps)
    def  getNeighbours(X,  point, eps):
        
        neighbors = []
        # for each point in the dataset
        for newPoint in range(len(X)):
            # if the distance is radius, add it to the neighbors list
            if np.linalg.norm(X[ point] - X[newPoint]) < eps:
                neighbors.append(newPoint)      
        return neighbors

# ## Function to Find eps value using kth Nearest Neighbors


from sklearn.neighbors import NearestNeighbors
# return best epsilon using NearestNeighbors
def find_optimum_eps(X):
    nbrs = NearestNeighbors(n_neighbors=5).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,4]
    # sort the distances and plotting K-distance Graph 
    plt.figure(figsize=(10,8))
    plt.plot(distances)
    plt.title('K-distance Graph',fontsize=10)
    plt.xlabel('Data Points sorted by distance',fontsize=10)
    plt.ylabel('Epsilon',fontsize=10)
    plt.show()

# ## Function to plot 


# return visualize for results using scatter plot
def plot(X,labal,title):
    plt.scatter(X[:,0],X[:,1], c=labal , cmap='Pastel1')
    plt.xlabel('******  X  *****', fontsize=14)
    plt.ylabel('******  Y  *****', fontsize=14) 
    plt.title(title)
    labal=pd.Series(labal)
    print('depending on %s the number of points in each cluster is:\n%s'%(title,labal.value_counts()))

# ## decideing GMM Componat


from sklearn import mixture
def GMM_copmonats(X):
    n_components = np.arange(1, 21)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
            for n in n_components]
    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')

# ## define global variable


# specified attributes(variables)
n_samples=300
raneemid=2+0+0+6+3+5+2
bedoorid=2+0+0+5+9+6+1
rifalid=2+0+0+6+7+5+8
random_state=475

# ### Genrate Dataset1: Blobs dataset


X,y = datasets.make_blobs(n_samples=n_samples,random_state=random_state)
plot(X,y,'real groubing')

# ### find optimumm value of k 


findoptimk(X)

# ### kmean fit Dataset1 


# callin kmean method and ploting
km=Kmeans(k=3,max_iter=10)
Kmean_pred_y =km.fit(X)
true_y=y
plot(X,Kmean_pred_y,'kmean grouping')

# ### find optimumm value of eps


find_optimum_eps(X)

# ### DBSCAN fit Dataset1


# callin dbscan method and plotting
DB_pred_y=dbscan(1.2, 5).fit(X)
true_y=y
plot(X,DB_pred_y,'DBSCAN grouping')

# ### Hirarical from sklearn data 1


from sklearn.cluster import AgglomerativeClustering
HC=AgglomerativeClustering(n_clusters=3, linkage = 'ward')
HC_pred_y=HC.fit_predict(X)
true_y=y
plot(X,HC_pred_y,'Hirarical grouping')



GMM_copmonats(X)

# ### GMM from sklearn data 1


gmm = GaussianMixture(n_components=3 , covariance_type='full', random_state=1)
GMM_pred_y= gmm.fit_predict(X)
true_y=y
plot(X,GMM_pred_y,'GMM grouping')

# # validation error 
# ON Dataset1


from sklearn.metrics import f1_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

# ### Kmean metrics 


#F-measure:
K_f1_D1=f1_score(true_y, Kmean_pred_y, average='macro')
K_f1_D1

#Rand index
K_R_D1=rand_score(true_y, Kmean_pred_y)
K_R_D1

#Normalized mutual information (NMI):
K_NMI_D1=normalized_mutual_info_score(true_y, Kmean_pred_y)
K_NMI_D1

# ### DBSCAN metrics


#F-measure:
DB_f1_D1=f1_score(true_y, DB_pred_y, average='macro')
DB_f1_D1

#Rand index
DB_R_D1=rand_score(true_y, DB_pred_y)
DB_R_D1

#Normalized mutual information (NMI):
DB_NMI_D1=normalized_mutual_info_score(true_y, DB_pred_y)
DB_NMI_D1

# ### Hirarical & GMM metrics


#Hirearcle
HC_f1_D1=f1_score(true_y, HC_pred_y, average='macro')
HC_R_D1=rand_score(true_y, HC_pred_y)
HC_NMI_D1=normalized_mutual_info_score(true_y, HC_pred_y)
#GMM
GM_f1_D1=f1_score(true_y, GMM_pred_y, average='macro')
GM_R_D1=rand_score(true_y, GMM_pred_y)
GM_NMI_D1=normalized_mutual_info_score(true_y, GMM_pred_y)

# ### Rankin Algorithm


#summary
table = [['K-mean',K_f1_D1, K_NMI_D1 , K_R_D1],  
         ['DBSACN',DB_f1_D1, DB_NMI_D1 ,DB_R_D1],
         ['HC',HC_f1_D1, HC_NMI_D1,GM_R_D1],
         ['GMM',GM_f1_D1, GM_NMI_D1,GM_R_D1]]
df_rank = pd.DataFrame(table, columns=['Models', 'F-measures','NMI','Rand Statistic'])
df_rank.set_index('Models')

# ranking models 
df_rank[['F-measures','NMI','Rand Statistic']] = df_rank[['F-measures','NMI','Rand Statistic']].rank(ascending=False)
df_rank['rank'] = df_rank.mean(axis=1)
df_rank['rank']=df_rank['rank'].rank(ascending=True)
df_rank

# ### Dataset2: Anisotropicly distributed dataset.


X, y = datasets.make_blobs(n_samples=n_samples,random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)
plot(X,y,'real groubing')

# ### find optimumm value of k 


# 


findoptimk(X)

# ### kmean fit Dataset2


# callin kmean method and ploting
km=Kmeans(k=3,max_iter=10)
Kmean_pred_y =km.fit(X)
true_y=y
plot(X,Kmean_pred_y,'kmean grouping')

# ### find optimumm value of eps


find_optimum_eps(X)

# ### DBSCAN fit Dataset2


# callin dbscan method and ploting
DB_pred_y=dbscan(0.7, 5).fit(X)
true_y=y
plot(X,DB_pred_y,'DBSCAN grouping')

# ### Hirarical from sklearn data 2


from sklearn.cluster import AgglomerativeClustering
HC=AgglomerativeClustering(n_clusters=3, linkage = 'ward')
HC_pred_y=HC.fit_predict(X)
true_y=y
plot(X,HC_pred_y,'Hirarical grouping')

GMM_copmonats(X)

# ### GMM from sklearn data 2


from sklearn import mixture
gmm = GaussianMixture(n_components=3 , covariance_type='full', random_state=1)
GMM_pred_y= gmm.fit_predict(X)
true_y=y
plot(X,GMM_pred_y,'GMM grouping')

# # validation error 
# KMean VS DBSCAN ON Dataset2


# ### Kmean metrics 


#F-measure:
K_f1_D2=f1_score(true_y, Kmean_pred_y, average='macro')
K_f1_D2

#Rand index
K_R_D2=rand_score(true_y, Kmean_pred_y)
K_R_D2

#Normalized mutual information (NMI):
K_NMI_D2=normalized_mutual_info_score(true_y, Kmean_pred_y)
K_NMI_D2

# ### DBSCAN metrics


#F-measure:
DB_f1_D2=f1_score(true_y, DB_pred_y, average='macro')
DB_f1_D2

#Rand index
DB_R_D2=rand_score(true_y, DB_pred_y)
DB_R_D2

#Normalized mutual information (NMI):
DB_NMI_D2=normalized_mutual_info_score(true_y, DB_pred_y)
DB_NMI_D2

# ### Hirarical & GMM metrics


#Hirearcle
HC_f1_D2=f1_score(true_y, HC_pred_y, average='macro')
HC_R_D2=rand_score(true_y, HC_pred_y)
HC_NMI_D2=normalized_mutual_info_score(true_y, HC_pred_y)
#GMM
GM_f1_D2=f1_score(true_y, GMM_pred_y, average='macro')
GM_R_D2=rand_score(true_y, GMM_pred_y)
GM_NMI_D2=normalized_mutual_info_score(true_y, GMM_pred_y)

# ### Rankin Algorithm


#summary
table = [['K-mean',K_f1_D2, K_NMI_D2 , K_R_D2],  
         ['DBSACN',DB_f1_D2, DB_NMI_D2 ,DB_R_D2],
         ['HC',HC_f1_D2, HC_R_D2,HC_NMI_D2],
         ['GMM',GM_f1_D2, GM_NMI_D2,GM_R_D2]]
df_rank = pd.DataFrame(table, columns=['Models', 'F-measures','NMI','Rand Statistic'])
df_rank.set_index('Models')

# ranking models 
df_rank[['F-measures','NMI','Rand Statistic']] = df_rank[['F-measures','NMI','Rand Statistic']].rank(ascending=False)
df_rank['rank'] = df_rank.mean(axis=1)
df_rank['rank']=df_rank['rank'].rank(ascending=True)
df_rank

# dubelicated 1.5 means that DBSCAN, GMM has same rank 1 
# while both KMean, Hierarchy ranked second


# ### Dataset3: Noisy moons dataset


X, y = datasets.make_moons(n_samples=n_samples, noise=0.1,random_state=random_state)
plot(X,y,'real groubing')

# ### find optimumm value of k 


findoptimk(X)

# ### kmean fit Dataset3


# callin kmean method and ploting
km=Kmeans(k=2,max_iter=10)
Kmean_pred_y =km.fit(X)
true_y=y
plot(X,Kmean_pred_y,'kmean grouping')

# ### find optimumm value of eps


find_optimum_eps(X)

# ### DBSCAN fit Dataset3


# callin dbscan method and ploting
DB_pred_y=dbscan(0.19, 5).fit(X)
true_y=y
plot(X,DB_pred_y,'DBSCAN grouping')

# ### Hirarical from sklearn data 3


from sklearn.cluster import AgglomerativeClustering
HC=AgglomerativeClustering(n_clusters=2, linkage = 'ward')
HC_pred_y=HC.fit_predict(X)
true_y=y
plot(X,HC_pred_y,'Hirarical grouping')

GMM_copmonats(X)

# ### GMM from sklearn data 3


from sklearn import mixture
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', random_state=1)
GMM_pred_y= gmm.predict(X)
true_y=y
plot(X,GMM_pred_y,'GMM grouping')

# # validation error 
# KMean VS DBSCAN ON Dataset3


# ### Kmean metrics 


#F-measure:
K_f1_D3=f1_score(true_y, Kmean_pred_y, average='macro')
K_f1_D3

#Rand index
K_R_D3=rand_score(true_y, Kmean_pred_y)
K_R_D3

#Normalized mutual information (NMI):
K_NMI_D3=normalized_mutual_info_score(true_y, Kmean_pred_y)
K_NMI_D3

# ### DBSCAN metrics


#F-measure:
DB_f1_D3=f1_score(true_y, DB_pred_y, average='macro')
DB_f1_D3

#Rand index
DB_R_D3=rand_score(true_y, DB_pred_y)
DB_R_D3

#Normalized mutual information (NMI):
DB_NMI_D3=normalized_mutual_info_score(true_y, DB_pred_y)
DB_NMI_D3

# ### Hirarical & GMM metrics


#Hirearcle
HC_f1_D3=f1_score(true_y, HC_pred_y, average='macro')
HC_R_D3=rand_score(true_y, HC_pred_y)
HC_NMI_D3=normalized_mutual_info_score(true_y, HC_pred_y)
#GMM
GM_f1_D3=f1_score(true_y, GMM_pred_y, average='macro')
GM_R_D3=rand_score(true_y, GMM_pred_y)
GM_NMI_D3=normalized_mutual_info_score(true_y, GMM_pred_y)

# ### Rankin Algorithm


#summary
table = [['K-mean',K_f1_D3, K_NMI_D3 , K_R_D3],  
         ['DBSACN',DB_f1_D3, DB_NMI_D3 ,DB_R_D3],
          ['HC',HC_f1_D3, HC_NMI_D3,GM_R_D3],
          ['GMM',GM_f1_D3, GM_NMI_D3,GM_R_D3]]
df_rank = pd.DataFrame(table, columns=['Models', 'F-measures','NMI','Rand Statistic'])
df_rank.set_index('Models')

# ranking models 
df_rank[['F-measures','NMI','Rand Statistic']] = df_rank[['F-measures','NMI','Rand Statistic']].rank(ascending=False)
df_rank['rank'] = df_rank.mean(axis=1)
df_rank['rank']=df_rank['rank'].rank(ascending=True)
df_rank

# ### Dataset4: noisy circles dataset


X,y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05,random_state=random_state)
plot(X,y,'real groubing')

# ### find optimumm value of k 


findoptimk(X)

# ### kmean fit Dataset4


# callin kmean method and ploting
km=Kmeans(k=2,max_iter=10)
Kmean_pred_y=km.fit(X)
true_y=y
plot(X,Kmean_pred_y,'kmean grouping')

# ### find optimumm value of eps


find_optimum_eps(X)

# ### DBSCAN fit Dataset4


# callin dbscan method and ploting
DB_pred_y=dbscan(0.18, 5).fit(X)
true_y=y
plot(X,DB_pred_y,'DBSCAN grouping')

# ### Hirarical from sklearn data 4


from sklearn.cluster import AgglomerativeClustering
HC=AgglomerativeClustering(n_clusters=2, linkage = 'ward')
HC_pred_y=HC.fit_predict(X)
true_y=y
plot(X,HC_pred_y,'Hirarical grouping')

GMM_copmonats(X)

# ### GMM from sklearn data 4


from sklearn import mixture
gmm = GaussianMixture(n_components=2 , covariance_type='full', random_state=1)
gmm.fit_predict(X)
GMM_pred_y= gmm.predict(X)
true_y=y
plot(X,GMM_pred_y,'GMM grouping')

# # validation error 
# KMean VS DBSCAN ON Dataset4


# ### Kmean metrics 


#F-measure:
K_f1_D4=f1_score(true_y, Kmean_pred_y, average='macro')
K_f1_D4

#Rand index
K_R_D4=rand_score(true_y, Kmean_pred_y)
K_R_D4

#Normalized mutual information (NMI):
K_NMI_D4=normalized_mutual_info_score(true_y, Kmean_pred_y)
K_NMI_D4

# ### DBSCAN metrics


#F-measure:
DB_f1_D4=f1_score(true_y, DB_pred_y, average='macro')
DB_f1_D4

#Rand index
DB_R_D4=rand_score(true_y, DB_pred_y)
DB_R_D4

#Normalized mutual information (NMI):
DB_NMI_D4=normalized_mutual_info_score(true_y, DB_pred_y)
DB_NMI_D4

# ### Hirarical & GMM metrics


#Hirearcle
HC_f1_D4=f1_score(true_y, HC_pred_y, average='macro')
HC_R_D4=rand_score(true_y, HC_pred_y)
HC_NMI_D4=normalized_mutual_info_score(true_y, HC_pred_y)
#GMM
GM_f1_D4=f1_score(true_y, GMM_pred_y, average='macro')
GM_R_D4=rand_score(true_y, GMM_pred_y)
GM_NMI_D4=normalized_mutual_info_score(true_y, GMM_pred_y)

# ### KMean VS DBSCAN 


#summary
table = [['K-mean',K_f1_D4, K_NMI_D4, K_R_D4],  
         ['DBSACN',DB_f1_D4, DB_NMI_D4,DB_R_D4],
         ['HC',HC_f1_D4, HC_R_D4,HC_NMI_D4],
         ['GMM',GM_f1_D4, GM_NMI_D4,GM_R_D4]]
df_rank = pd.DataFrame(table, columns=['Models', 'F-measures','NMI','Rand Statistic'])
df_rank.set_index('Models')

# ### Rankin Algorithm


# ranking models 
df_rank[['F-measures','NMI','Rand Statistic']] = df_rank[['F-measures','NMI','Rand Statistic']].rank(ascending=False)
df_rank['rank'] = df_rank.mean(axis=1)
df_rank['rank']=df_rank['rank'].rank(ascending=True)
df_rank

# # Analysis 📈
# 
# On dataset 1 KMean ranked first after that GMM, dataset 2 GMM and DBSCAN ranked first, dataset 3 DBSCAN first second HC in dataset 4 DBSCAN, F-measure and NMI DBSCAN gave higher results closer to +1 which is the best.


# # Conclusion 📑
# DBSCAN may not be the most simple or well-known clustering algorithm, but it has its advantages. We have shown how K Means can generate clusters that do not make sense sometimes, while GMM strength that it dose not require k also DBSCAN can capture the dense regions and identify noise. Hirearcle allow overlaping. But, the final decision is based on your data type and which will benefit you the most.


# # Reference 🗂️
# 1- Ali, M. (2021, November 29). DBSCAN Clustering Algorithm Implementation from scratch | Python. In Medium. https://becominghuman.ai/dbscan-clustering-algorithm-implementation-from-scratch-python-9950af5eed97
# In-Text Citation: (Ali, 2021).
# 
# 2- dr.Elham, Algamdi. (2023, January 16). notebooks KMean_base Clustering Algorithm from scratch | Python.


# # Authorized by🏅
# 
# **Raneem Alomari 2006352**
# 
# **Bedoor Aayd 2005961**
# 
# **Rifal Almaghrabi 2006758** 


