"""
@author: Rohit Pawar
Project 2: Study of K-Means in Image Classification
"""

import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from plotly.graph_objs import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import fetch_mldata
from time import time	
from __future__ import print_function
import matplotlib.pyplot as plt
import os
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn import metrics
from sklearn.cluster import KMeans	
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import spectral_clustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.tools as tls

# plotly library credentials.
tls.set_credentials_file(username='*********', api_key='*************')


mnistDataSet = fetch_mldata('MNIST original')
mnistComponents = 10
mnistSampleCount = mnistDataSet.target.shape
mnistFeatureCount = mnistData[0].shape
mnistClasses = np.unique(mnistDataSet.target)
print("Total MNIST dataset size:")
print("Samples: %d" %mnistSampleCount)
print("Features: %d" %mnistFeatureCount)
print("Classes: %d" %mnistClasses.shape)
'''
Total MNIST dataset size:
Samples: 70000
Features: 784
Classes: 10
'''
'''
    MNIST original
'''

mnistData = mnistDataSet.data
mnistTarget = mnistDataSet.target
target_names = mnistDataSet.COL_NAMES

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

X_train, X_test, y_train, y_test = train_test_split(mnistData, mnistTarget, test_size=0.642, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
print("Total MNIST dataset used size:")
print("Training Samples: %d" %y_train.shape)
print("Testing Samples: %d" %y_test.shape)

'''
Total MNIST dataset used size:
Training Samples: 20048
Testing Samples: 5012
'''



#STORE MNIST Data OUTPUT on DISK
pickle_file = os.path.join(".", 'MNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
     'train_dataset':X_train,
     'train_labels':y_train,
     'test_dataset':X_test,
     'test_labels':y_test
    }
  pickle.dump(save, f, 3)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise  

MNISTPickleDS= pickle.load( open( "MNIST.pickle", "rb" ) )


X_train = MNISTPickleDS['train_dataset']
y_train = MNISTPickleDS['train_labels']
X_test = MNISTPickleDS['test_dataset']
y_test = MNISTPickleDS['test_labels']

MNIST_data=np.concatenate((X_train, X_test), axis=0)
MNIST_labels=np.concatenate((y_train, y_test), axis=0)

#IMPLEMETING K-MEANS WITH MULTIPLE CLUSTERS AND CHECK PERFORMANCE

KMeansRange = []

for index,n_cluster in enumerate(range(2,21,2)):
    kmeans = KMeans(n_clusters=n_cluster, n_init=1)
    kmeans.fit(X_train)
    labels = kmeans.labels_
    KMeansOutputAndPerformance = []
    KMeansOutputAndPerformance.append(n_cluster)
    KMeansOutputAndPerformance.append(labels)
    KMeansOutputAndPerformance.append([metrics.adjusted_rand_score(y_train, labels)])
    KMeansOutputAndPerformance.append([metrics.v_measure_score(y_train, labels)])
    KMeansOutputAndPerformance.append([metrics.silhouette_score(X_train, labels, metric='euclidean')])
    KMeansRange.append(KMeansOutputAndPerformance)

#STORE KMeans MNIST RangeOutput OUTPUT on DISK
pickle_file = os.path.join(".", 'KMeans_MNIST_RangeOutput.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'KMeansRange': KMeansRange
    }
  pickle.dump(save, f, 3)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise  

KMeansRangeOutput= pickle.load( open( "KMeans_MNIST_RangeOutput.pickle", "rb" ) )
KMeansRange = KMeansRangeOutput['KMeansRange']
KMeansRange = np.array(KMeansRange, dtype=object)  

#PLOTS OF SCORES

def getMedian(list):
    medians = []
    for i in list:
        medians.append(np.median(i))
    return medians

scoreNames=['adjusted_rand','v_measure','silhouette']

#MatlabPlot for KMeans Score
'''
plots = []
for score in range(len(scoreNames)):
    plots.append(plt.errorbar(range(2,21,2), getMedian(KMeansRange[:,score+2]), np.std(KMeansRange[score][score+2]),capsize=1))
    plt.title("Clustering Scores on range of clusters")
    plt.xlabel('Number of clusters')
    plt.ylabel('Score value')
    plt.ylim(ymin=-0.05, ymax=1.05)
    plt.legend(plots, scoreNames)
    plt.legend(scoreNames,loc='upper right')
plt.show()
'''

#Plotly for KMeans Score
trace1 = go.Scatter(
    x=[i for i in range(2,21,2)],
    y=getMedian(KMeansRange[:,2]),
    mode='lines+markers',
    name='adjusted_rand Scores',
    hoverinfo='name',
    line=dict(
        shape='spline'
    )
)
trace2 = go.Scatter(
    x=[i for i in range(2,21,2)],
    y=getMedian(KMeansRange[:,3]),
    mode='lines+markers',
    name='v_measure Scores',
    hoverinfo='name',
    line=dict(
        shape='spline'
    )
)
trace3 = go.Scatter(
    x=[i for i in range(2,21,2)],
    y=getMedian(KMeansRange[:,4]),
    mode='lines+markers',
    name='silhouette Scores',
    hoverinfo='name',
    line=dict(
        shape='spline'
    )
)
data = [trace1,trace2,trace3]
layout = dict(
    yaxis=YAxis(title='Score value'),
    xaxis=XAxis(title='Number of Clusters'),
    title= 'Score values by number of clusters',
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Clustering-Scores-MNIST')

CLUSTER_COUNT = 8

#DID NOT WORK FOR 1.5 Hrs
''' 
#Eigendecomposition of the raw data based on the correlation matrix:
correlationMatrix = np.corrcoef(X_train)
pcaEigenVals, pcaEigenVecs = np.linalg.eig(correlationMatrix)
'''

PCA_COUNT = 20

#LogisticRegression
totalTime = time()
# Instantiate
lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)
# Fit
scores = cross_val_score(lg, MNIST_data, MNIST_labels, cv=5)
#lg.fit(X_train, y_train)
#print('Total LogisticRegression Training Time :', time()-totalTime)

#totalTime = time()
# Predict
#y_pred = lg.predict(X_test)
print("Time for LogisticRegression Predict: ",time()-totalTime)

# Score
#score = metrics.accuracy_score(y_test, y_pred)
#print("Overall Accuracy LogisticRegression: ", score)
print("Overall Accuracy LogisticRegression: %0.2f (+/- %0.2f)" %(scores.mean(),scores.std()*2))

#Complete
'''
Total LogisticRegression Training Time : 201.9549560546875
Time for LogisticRegression Predict:  0.030582189559936523
Total Time: 201.98553824424744
Overall Accuracy LogisticRegression:  0.868116520351
'''
#Cross Validation
'''

'''


# KMEANS + LogisticRegression 
n_digits = 10
#reduced_data = PCA(n_components=n_digits).fit_transform(X_train)
kmeans = KMeans(n_clusters=10, n_init=1)
kmeans.fit(X_train)
#reduced_testdata = PCA(n_components=n_digits).fit_transform(X_test)
# Get Data and Lables by Cluster
LRModels = []

tLRStartTime = time()

for i in range(kmeans.n_clusters):
	t0 = time()
	clusterIndices = np.where(kmeans.labels_ == i)
	lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)
    # Fit
	lg.fit(X_train[clusterIndices], y_train[clusterIndices])
	LRModels.append(lg)

print('Total LR Training Time :', time()-tLRStartTime)

from sklearn import metrics
testClusters = kmeans.predict(X_test)
totalScore = 0
tLRStartTime = time()
for i in range(kmeans.n_clusters):
	t0 = time()
	clusterIndices = np.where(testClusters == i)
	y_pred = LRModels[i].predict(X_test[clusterIndices])
	score =metrics.accuracy_score(y_test[clusterIndices], y_pred)
	totalScore = totalScore + score
print("Time for LR Predict: ",time()-tLRStartTime)
print("Overall Accuracy LogisticRegression: ", str(totalScore/kmeans.n_clusters))

#KMeans Clusters = 8 + LogisticRegression without PCA
'''
Total LR Training Time : 295.44145607948303
Time for LR Predict:  0.035092830657958984
Overall Accuracy LogisticRegression:  0.911276988502
'''

#KMEANS + LogisticRegression without PCA
'''
Total LR Training Time : 88.6462972164154
Time for LR Predict:  0.06860089302062988
Overall Accuracy LogisticRegression:  0.935981125011
'''
#KMEANS + LogisticRegression with PCA
'''
Total LR Training Time : 22.42997407913208
Time for LR Predict:  0.046884775161743164
Overall Accuracy LogisticRegression:  0.167514976692
'''


KMeans_LR = np.array([['LogisticRegression Complete',201.98553824424744,0.868116520351],
             ['KMeans + LogisticRegression',88.71489810943604,0.935981125011],
             ['PCA + KMeans + LogisticRegression',22.476858854293823,0.167514976692]])

#PlotLy Bar PLOT
#Accuracy-by-LogisticRegression-Pipeline
trace = Bar(
        x= [name for name in KMeans_LR[:,0]],
        y=[round(float(accuracy)*100,2) for accuracy in KMeans_LR[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in KMeans_LR[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy by LogisticRegression Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Accuracy-by-LogisticRegression-Pipeline')

#Computation-Time-by-LogisticRegression-Pipeline
trace = Bar(
        x= [name for name in KMeans_LR[:,0]],
        y=[round(float(time),2) for time in KMeans_LR[:,1]],
        text=[round(float(time),2) for time in KMeans_LR[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time by LogisticRegression Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Computation-Time-by-LogisticRegression-Pipeline')


#Plot with Matplot
'''
y_pos = np.arange(len(KMeans_LR[:,0]))
bar_width = 0.35
plt.bar(y_pos, [float(accuracy)*100 for accuracy in KMeans_LR[:,2]] , align = 'center',alpha = 0.5)
                 #color='b',
                 #label=[name for name in enumerate(KMeans_LR[:,0])])
plt.xlabel('Methods')
plt.ylabel('Accuracy %')
plt.title('Accuracy by Methods')
plt.xticks(y_pos, [name for name in KMeans_LR[:,0]])
plt.legend()

plt.tight_layout()
plt.show()
'''


#SVM COMPLETE

totalTime = time()
clf = svm.SVC()
#clf.fit(X_train, y_train)  
scores = cross_val_score(clf, MNIST_data, MNIST_labels, cv=5)
print("SVM Cross Validation Total Time :", (time()-totalTime))
print("Overall Accuracy:  %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
SVM Cross Validation Total Time : 4952.866276741028
Overall Accuracy:  0.11 (+/- 0.00)
'''

# KMeans + SVM

n_digits = 10
reduced_data = PCA(n_components=n_digits).fit_transform(X_train)
kmeans = KMeans(n_clusters=10, n_init=1)
kmeans.fit(reduced_data)
reduced_testdata = PCA(n_components=n_digits).fit_transform(X_test)
# Get Data and Lables by Cluster
SVMModels = []
totalTime = time()
for i in range(kmeans.n_clusters):
    clusterIndices = np.where(kmeans.labels_ == i)
    clf = svm.SVC()
    clf.fit(reduced_data[clusterIndices], y_train[clusterIndices])
    SVMModels.append(clf)
print('SVM TRAIN TIME: ',str(time()-totalTime))

totalTime = time()
from sklearn import metrics
testClusters = kmeans.predict(reduced_testdata)
totalScore = 0
for i in range(kmeans.n_clusters):
    t0 = time()
    clusterIndices = np.where(testClusters == i)
    score =SVMModels[i].score(reduced_testdata[clusterIndices],y_test[clusterIndices])
    totalScore = totalScore + score
print('SVM SCORE TIME: ',str(time()-totalTime))
print("Overall Accuracy is :", str(totalScore/kmeans.n_clusters))

#Kmeans + SVM without PCA
'''
SVM TRAIN TIME:  235.4332194328308
SVM SCORE TIME:  13.405681610107422
Overall Accuracy is : 0.651705449289
'''
#Kmeans + SVM with PCA
'''
SVM TRAIN TIME:  11.965496301651001
SVM SCORE TIME:  0.6922013759613037
Overall Accuracy is : 0.642165395065
'''

KMeans_SVM = np.array([['SVM Complete', float(4952.866276741028/5), 0.11],
             ['KMeans + SVM', 248.83890104293823, 0.651705449289],
             ['PCA + KMeans + SVM', 12.657697677612305, 0.642165395065]])

#PlotLy Bar PLOT
#Accuracy-by-SVM-Pipeline
trace = Bar(
        x= [name for name in KMeans_SVM[:,0]],
        y=[round(float(accuracy)*100,2) for accuracy in KMeans_SVM[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in KMeans_SVM[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy by SVM Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Accuracy-by-SVM-Pipeline')

#Computation-Time-by-SVM-Pipeline
trace = Bar(
        x= [name for name in KMeans_SVM[:,0]],
        y=[round(float(time),2) for time in KMeans_SVM[:,1]],
        text=[round(float(time),2) for time in KMeans_SVM[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time by SVM Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Computation-Time-by-SVM-Pipeline')



#DecisionTree COMPLETE
totalTime = time()
clf = tree.DecisionTreeClassifier()
#clf.fit(X_train, y_train)
#print('DT TRAIN TIME: ',str(time()-totalTime))
#totalTime = time()
#y_predict =clf.predict(X_test)
#score =metrics.accuracy_score(y_test, y_predict)
scores = cross_val_score(clf, MNIST_data, MNIST_labels, cv=5)
print('DT Cross Validation Total TIME: ',str(time()-totalTime))
#print("Overall Accuracy is :", str(score))
print("Overall Accuracy DecisionTree: %0.2f (+/- %0.2f)" %(score.mean(),scores.std()*2))


#Output DT CrossValidation
'''
DT Cross Validation Total TIME:  30.64122986793518
Overall Accuracy DecisionTree: 0.92 (+/- 0.00)
'''

#OUTPUT DT COMPLETE
'''
DT TRAIN TIME:  6.168358564376831
Overall Accuracy is : 0.837589784517
'''

# KMEANS + DecisionTree

reduced_data = PCA(n_components=PCA_COUNT).fit_transform(X_train)
kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=1)
kmeans.fit(reduced_data)
reduced_testdata = PCA(n_components=PCA_COUNT).fit_transform(X_test)
# Get Data and Lables by Cluster
DTModels = []
totalTime = time()
for i in range(kmeans.n_clusters):
    clusterIndices = np.where(kmeans.labels_ == i)
    clf = tree.DecisionTreeClassifier()
    clf.fit(reduced_data[clusterIndices], y_train[clusterIndices])
    DTModels.append(clf)
print('DT TRAIN TIME: ',str(time()-totalTime))

totalTime = time()
from sklearn import metrics
testClusters = kmeans.predict(reduced_testdata)
totalScore = 0
for i in range(kmeans.n_clusters):
    t0 = time()
    clusterIndices = np.where(testClusters == i)
    y_predict =DTModels[i].predict(reduced_testdata[clusterIndices])
    score =metrics.accuracy_score(y_test[clusterIndices], y_predict)
    totalScore = totalScore + score
print('DT SCORE TIME: ',str(time()-totalTime))
print("Overall Accuracy is :", str(totalScore/kmeans.n_clusters))

#OUTPUT WITHOUT PCA
'''
DT TRAIN TIME:  4.934544801712036
DT SCORE TIME:  0.05498838424682617
Overall Accuracy is : 0.881269617727
'''

#OUTPUT WITH PCA
'''
DT TRAIN TIME:  0.8034610748291016
DT SCORE TIME:  0.00800013542175293
Overall Accuracy is : 0.175886522807
'''

KMeans_DT = np.array([['DT Complete', 6.168358564376831, 0.837589784517],
             ['KMeans + DT',  4.989533185958862, 0.881269617727],
             ['PCA + KMeans + DT', 0.8114612102508545, 0.175886522807]])

#PlotLy Bar PLOT
#Accuracy-by-DecisionTree-Pipeline
trace = Bar(
        x= [name for name in KMeans_DT[:,0]],
        y=[round(float(accuracy)*100,2) for accuracy in KMeans_DT[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in KMeans_DT[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy by DecisionTree Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Accuracy-by-DecisionTree-Pipeline')

#Computation-Time-by-DecisionTree-Pipeline
trace = Bar(
        x= [name for name in KMeans_DT[:,0]],
        y=[round(float(time),2) for time in KMeans_DT[:,1]],
        text=[round(float(time),2) for time in KMeans_DT[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time by DecisionTree Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Computation-Time-by-DecisionTree-Pipeline')


#KNeighborsClassifier Complete
totalTime = time()
kNC = KNeighborsClassifier(n_neighbors=3)
#kNC.fit(X_train,y_train)
#print('KNeighborsClassifier Complete TRAIN TIME: ',str(time()-totalTime))
scores = cross_val_score(kNC, MNIST_data, MNIST_labels, cv=5)
#totalTime = time()
#score =kNC.score(X_test,y_test)
#print('KNeighborsClassifier Complete SCORE TIME: ',str(time()-totalTime))
#print("KNeighborsClassifier Complete Accuracy is :", score)
print('KNeighborsClassifier Complete Cross Validation Total TIME: ',str(time()-totalTime))
print("Overall Accuracy KNeighborsClassifier Complete: %0.2f (+/- %0.2f)" %(score.mean(),scores.std()*2))

#Output Complete

#Output Cross Validation
'''
KNeighborsClassifier Complete TRAIN TIME:  511.0815701484680176
KNeighborsClassifier Complete Accuracy is : 0.964086193136
'''

# KMEANS + KNeighborsClassifier 

n_digits = 20
#reduced_data = PCA(n_components=n_digits).fit_transform(X_train)
kmeans = KMeans(n_clusters=10, n_init=1)
kmeans.fit(X_train)
reduced_testdata = PCA(n_components=n_digits).fit_transform(X_test)
# Get Data and Lables by Cluster
KNCModels = []
totalTime = time()
for i in range(kmeans.n_clusters):
    clusterIndices = np.where(kmeans.labels_ == i)
    kNC = KNeighborsClassifier(n_neighbors=3)
    kNC.fit(X_train[clusterIndices],y_train[clusterIndices])
    KNCModels.append(kNC)
print('kNC TRAIN TIME: ',str(time()-totalTime))

totalTime = time()
from sklearn import metrics
testClusters = kmeans.predict(X_test)
totalScore = 0
for i in range(kmeans.n_clusters):
    t0 = time()
    clusterIndices = np.where(testClusters == i)
    score =KNCModels[i].score(X_test[clusterIndices],y_test[clusterIndices])
    totalScore = totalScore + score
print('kNC SCORE TIME: ',str(time()-totalTime))
print("Overall Accuracy is :", str(totalScore/kmeans.n_clusters))

#Without PCA
'''
kNC TRAIN TIME:  0.6958816051483154
kNC SCORE TIME:  17.866215705871582
Overall Accuracy is : 0.950871974738
'''

#With PCA
'''
kNC TRAIN TIME:  0.04050755500793457
kNC SCORE TIME:  0.562324047088623
Overall Accuracy is : 0.169958520944
'''

KMeans_kNC = np.array([['KNeighbors Complete', float(511.1551651954651/5), 0.964086193136],
             ['KMeans + KNeighbors', 18.562097311019897, 0.950871974738],
             ['PCA + KMeans + KNeighbors', 0.6028316020965576, 0.169958520944]])

#PlotLy Bar PLOT
#Accuracy-by-KNeighbors-Pipeline
trace = Bar(
        x= [name for name in KMeans_kNC[:,0]],
        y=[round(float(accuracy)*100,2) for accuracy in KMeans_kNC[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in KMeans_kNC[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy by KNeighbors Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Accuracy-by-KNeighbors-Pipeline')

#Computation-Time-by-KNeighbors-Pipeline
trace = Bar(
        x= [name for name in KMeans_kNC[:,0]],
        y=[round(float(time),2) for time in KMeans_kNC[:,1]],
        text=[round(float(time),2) for time in KMeans_kNC[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time by KNeighbors Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Computation-Time-by-KNeighbors-Pipeline')



#PlotLy Bar PLOT for all Accuracy
KMeans_MNIST = np.column_stack((np.column_stack((np.column_stack((KMeans_LR,KMeans_SVM)),KMeans_DT)),KMeans_kNC))

trace1 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        y=[round(float(accuracy)*100,2) for accuracy in KMeans_LR[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in KMeans_LR[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)),
        name="Logistic Regression")

trace2 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="SVM",
        y=[round(float(accuracy)*100,2) for accuracy in KMeans_SVM[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in KMeans_SVM[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))


trace3 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="Decision Tree",
        y=[round(float(accuracy)*100,2) for accuracy in KMeans_DT[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in KMeans_DT[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

trace4 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="KNeighbors Classifier",
        y=[round(float(accuracy)*100,2) for accuracy in KMeans_kNC[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in KMeans_kNC[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))


data = Data([trace1,trace2,trace3,trace4])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy for MNIST Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Accuracy-for-MNIST-Pipeline')


#PlotLy Bar PLOT for all Computation-Time-
#KMeans_MNIST = np.column_stack((np.column_stack((np.column_stack((KMeans_LR,KMeans_SVM)),KMeans_DT)),KMeans_kNC))

trace1 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        y=[round(float(accuracy),2) for accuracy in KMeans_LR[:,1]],
        text=[round(float(accuracy),2) for accuracy in KMeans_LR[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)),
        name="Logistic Regression")

trace2 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="SVM",
        y=[round(float(accuracy),2) for accuracy in KMeans_SVM[:,1]],
        text=[round(float(accuracy),2) for accuracy in KMeans_SVM[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))


trace3 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="Decision Tree",
        y=[round(float(accuracy),2) for accuracy in KMeans_DT[:,1]],
        text=[round(float(accuracy),2) for accuracy in KMeans_DT[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

trace4 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="KNeighbors Classifier",
        y=[round(float(accuracy),2) for accuracy in KMeans_kNC[:,1]],
        text=[round(float(accuracy),2) for accuracy in KMeans_kNC[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))


data = Data([trace1,trace2,trace3,trace4])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time for MNIST Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Computation-Time-by-MNIST-Pipeline')