# K-mean-and-DBSCAN-from-scratch

This was a project created as part of a CCAI-323 (Machine Learning)

Most of the traditional clustering techniques, such as k-means and DBSCAN clustering, can be used to group data without supervision. In this project, we will implement two clustering algorithms from scratch and try it on a customized dataset.

Finally, we will check the quality of each clustering using external index to compare between algorithms.

# Objective
This project aims to demonstrate the ability to apply Machine Learning algorithms to analyze different scenarios, translate theory into practice, apply unsupervised learning techniques, and analyze output for better understanding and decision making.

# Classes are:
1-  KMean class

            - fit function 
            
            - update centroid
            
            - assaign cluster
            
2-  DBSCAN class

            - fit
            
            - get neighbors 
            
 # Functions are:
    1- elbow function
    
    2- nearest neighbors function
    
    3- ploting function
    
    
# Validation measures:
    1- F-measure
    
    2- Rand statics
    
    3- Normlized matual information 
# dataset:
    1- Blobs dataset
    
![image](https://user-images.githubusercontent.com/97242283/225445446-cff5a9cb-b243-45cd-9354-153ef5d58ce7.png)

    2- Anisotropicly distributed dataset.
 
 ![image](https://user-images.githubusercontent.com/97242283/225445701-e4d29afc-2a29-42f2-9a16-3f622bb9187f.png)
   
    3-  Noisy moons dataset

![image](https://user-images.githubusercontent.com/97242283/225446364-e3a403d8-829b-4139-a732-d2fec79423b3.png)

    4-  noisy circles dataset
    
![image](https://user-images.githubusercontent.com/97242283/225446137-227985d6-1e93-463e-9f66-314cb184af7c.png)

# kmean fit
### Blobs dataset

![image](https://user-images.githubusercontent.com/97242283/225447325-425b4767-6957-4c00-aa89-d56f9c41524e.png)

### Anisotropicly distributed dataset.

![image](https://user-images.githubusercontent.com/97242283/225447401-06a4757e-4d00-4663-8a7c-414d25adc67b.png)

### Noisy moons dataset

![image](https://user-images.githubusercontent.com/97242283/225447468-94db9443-a9b8-439f-97bb-5a52a551d613.png)

### noisy circles dataset

![image](https://user-images.githubusercontent.com/97242283/225447561-109c29ef-b9dc-497e-bcb9-ad85b5a48ac4.png)

# DBSCAN fit
### Blobs dataset

![image](https://user-images.githubusercontent.com/97242283/225447778-41226418-1d41-4720-ad81-1af0a787097e.png)

### Anisotropicly distributed dataset.

![image](https://user-images.githubusercontent.com/97242283/225447844-0aa30c18-f417-488e-95ed-8350ad41eb77.png)

### Noisy moons dataset

![image](https://user-images.githubusercontent.com/97242283/225447887-63d2074c-b21c-41b7-ad7b-330f0e68675e.png)

# measures values:
### Blobs dataset

![image](https://user-images.githubusercontent.com/97242283/225448497-045b2f76-6622-4e02-9f5a-63b4b16c7ca2.png)

### Anisotropicly distributed dataset.

![image](https://user-images.githubusercontent.com/97242283/225448646-f22800e6-7e5b-45fd-9965-be295a486e96.png)

### Noisy moons dataset

![image](https://user-images.githubusercontent.com/97242283/225448717-565bb243-3b6d-4429-9f46-f3680d223332.png)

### noisy circles dataset

![image](https://user-images.githubusercontent.com/97242283/225449254-30e5fa13-0042-42d7-92e9-cfef3f82db8e.png)

# Analysis
On dataset 1 KMean ranked first, dataset 2  DBSCAN ranked first, dataset 3 DBSCAN first in dataset 4 DBSCAN.
