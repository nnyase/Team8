import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

def cluster(pathFolder, plot = False):

    """ This function compute K means with K = 5 for painting images in the database according to their color (HSV). 
    
    Parameters
    ----------
    pathFolder : string
        Path of the folder where the paintings image are stored.
    plot : boolean
        Plot the cluster
    
    Returns
    -------    
    """
        
    # Sorted the images
    files = sorted(os.listdir(pathFolder))
    result = []
        
    # For each image in the dataset
    for i, fileQ in enumerate(files):

            # Load the image
            image = cv2.imread(pathFolder + fileQ)
            # Convert it into HSV
            hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h = np.median(hsvImage[:,:,0])
            s = np.median(hsvImage[:,:,1])
            v = np.median(hsvImage[:,:,2])
            # Store the mean value of each HSV channel into result
            result.append([h,s,v])

    # Convert result list into an numpy array
    X = np.array(result)
    # Instantiate Kmeans
    km = KMeans(n_clusters=5).fit(X)
        
    print('Correspond cluster for each image')
    clusts = km.fit_predict(X)
    print(clusts)
    
    print('Best correspondance image for each cluster')
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    print(closest)

    if plot == True:
        clusts = km.fit_predict(X)
        #Plot the clusters obtained using k means
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(km.cluster_centers_[:, 0],
                    km.cluster_centers_[:, 1],
                    km.cluster_centers_[:, 2],
                    s = 250,
                    marker='o',
                    c='red',
                    label='centroids')
        scatter = ax.scatter(X[:,0],X[:,1],X[:,2],
                            c=clusts,s=20, cmap='winter')

        ax.set_title('K-Means Clustering')
        ax.set_xlabel('Hue')
        ax.set_ylabel('Saturation')
        ax.set_zlabel('Value')
        ax.legend()
        plt.show()


def new_cluster(pathFolder, sadImagePath, happyImagePath, zenImagePath, crazyImagePath, nostalgicImagePath):

    """ This function compute for each image in the database its cluster's correspondance. Cluster correspond 
    respectively to the feeling sad, happy, zen, crazy and nostalgic. We get the HSV value of each of this feeling
    by chosing one painting correspond to this emotion.
    
    Parameters
    ----------
    pathFolder : string
        Path of the folder where the paintings image are stored.
    sadImagePath : string
        Path of the image with sad emotion
    happyImagePath : string 
        Path of the image with happy emotion
    zenImagePath : string
        Path of the image with zen emotion
    crazyImagePath : string
        Path of the image with crazy emotion
    nostalgicImagePath : string
        Path of the image with nostalgic emotion
    Returns
    -------   
    finals : np array containing for each image its cluster number and distances from it 
    """


    # Sad image
    sadImage = cv2.imread(sadImagePath)
    sadHSVImage = cv2.cvtColor(sadImage, cv2.COLOR_BGR2HSV)
    sadVector = np.array( [np.median(sadHSVImage[:,:,0]), np.median(sadHSVImage[:,:,1]), np.median(sadHSVImage[:,:,2])] )

     # Happy image
    happyImage = cv2.imread(happyImagePath)
    happyHSVImage = cv2.cvtColor(happyImage, cv2.COLOR_BGR2HSV)
    happyVector = np.array( [np.median(happyHSVImage[:,:,0]), np.median(happyHSVImage[:,:,1]), np.median(happyHSVImage[:,:,2])] )

     # Zen image
    zenImage = cv2.imread(zenImagePath)
    zenHSVImage = cv2.cvtColor(zenImage, cv2.COLOR_BGR2HSV)
    zenVector = np.array( [np.median(zenHSVImage[:,:,0]), np.median(zenHSVImage[:,:,1]), np.median(zenHSVImage[:,:,2])] )

     # Crazy image
    crazyImage = cv2.imread(crazyImagePath)
    crazyHSVImage = cv2.cvtColor(crazyImage, cv2.COLOR_BGR2HSV)
    crazyVector = np.array( [np.median(crazyHSVImage[:,:,0]), np.median(crazyHSVImage[:,:,1]), np.median(crazyHSVImage[:,:,2])] )

     # Sad image
    nostalgicImage = cv2.imread(nostalgicImagePath)
    nostalgicHSVImage = cv2.cvtColor(nostalgicImage, cv2.COLOR_BGR2HSV)
    nostalgicVector = np.array( [np.median(nostalgicHSVImage[:,:,0]), np.median(nostalgicHSVImage[:,:,1]), np.median(nostalgicHSVImage[:,:,2])] )

    # Sorted the images
    files = sorted(os.listdir(pathFolder))
    result = []
        
    # For each image in the dataset
    for i, fileQ in enumerate(files):

            # Load the image
            image = cv2.imread(pathFolder + fileQ)
            # Convert it into HSV
            hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h,s,v = np.median(hsvImage[:,:,0]), np.median(hsvImage[:,:,1]), np.median(hsvImage[:,:,2])
            # Store the mean value of each HSV channel into result
            result.append([h,s,v])

    final = []
    
    for i in range(len(result)):

        distSad = np.linalg.norm(result[i] - sadVector)
        distHappy = np.linalg.norm(result[i] - happyVector)
        distZen = np.linalg.norm(result[i] - zenVector)
        distCrazy = np.linalg.norm(result[i] - crazyVector)
        distNostalgic = np.linalg.norm(result[i] - nostalgicVector)

        distances = (distSad,distHappy,distZen,distCrazy,distNostalgic)
        final.append([distances.index(min(distances)),min(distances)])

    final = np.array(final)
    return final

# result = new_cluster('BBDD/',sadImagePath = 'BBDD/bbdd_00009.jpg',
# happyImagePath = 'BBDD/bbdd_00167.jpg', zenImagePath = 'BBDD/bbdd_00047.jpg',
# crazyImagePath = 'BBDD/bbdd_00086.jpg', nostalgicImagePath = 'BBDD/bbdd_00078.jpg')

# print(np.where(result[:,0] == 4))






