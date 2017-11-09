"""
Study of K-Means in Image Classification
@author: Rohit Pawar
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans	
from time import time		
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import plotly.tools as tls
import plotly.graph_objs as go


# plotly library credentials.
tls.set_credentials_file(username='rohitpawar.social', api_key='IocxBcAIfahQXPOeZfWM')

#os.chdir('***********')

notMNISTPickleDS= pickle.load( open( "notMNIST.pickle", "rb" ) )


train_dataset = notMNISTPickleDS['train_dataset']
train_labels = notMNISTPickleDS['train_labels']
valid_dataset = notMNISTPickleDS['valid_dataset']
valid_labels  = notMNISTPickleDS['valid_labels']
test_dataset = notMNISTPickleDS['test_dataset']
test_labels = notMNISTPickleDS['test_labels']

# Prepare training data
samples, width, height = train_dataset.shape
X_train = np.reshape(train_dataset,(samples,width*height))
y_train = train_labels

# Prepare testing data
samples, width, height = test_dataset.shape
X_test = np.reshape(test_dataset,(samples,width*height))
y_test = test_labels

labels = y_train
data = X_train

NonMNIST_data = np.concatenate((X_train,X_test),axis=0)
NonMNIST_test = np.concatenate((y_train,y_test),axis=0)

print("Total NonMNIST dataset used size:")
print("Training Samples: %d" %y_train.shape)
print("Testing Samples: %d" %y_test.shape)

'''
Total NonMNIST dataset used size:
Training Samples: 50000
Testing Samples: 5000
'''

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


KMeansRangeOutput= pickle.load( open( "KMeans_NONMNIST_RangeOutput.pickle", "rb" ) )
KMeansRange = KMeansRangeOutput['KMeansRange']
KMeansRange = np.array(KMeansRange, dtype=object)

#STORE KMEANS MNIST OUTPUT on DISK
pickle_file = os.path.join(".", 'KMeans_NONMNIST_RangeOutput.pickle')

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
    yaxis=YAxis(title='Score'),
    xaxis=XAxis(title='Number of Clusters'),
    title= 'Score By Clusters',
    )
fig = go.Figure(data=data, layout=layout)
#fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Clustering-Scores-NonMNIST')



CLUSTER_COUNT = 10
#Implementing KMEANS (CLUSTER_COUNT=10) + SVM over multiple count of PC's

NonMNIST_svm_PerPC = np.zeros(shape=(1,4))

for count_PC in range(10,210,10):
    #APPLY PCA
    totalTime = time()
    svmOutput = np.array([])

    svmOutput = np.append(svmOutput,count_PC) #Store PC Count
    reduced_data = PCA(n_components=count_PC).fit_transform(data)
    reduced_testdata = PCA(n_components=count_PC).fit_transform(X_test)
    #APPLY KMEANS
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=1)
    kmeans.fit(reduced_data)
    print('PCA Complete : ',count_PC,(time()-totalTime))
    # Get Data and Lables by Cluster
    SVMModels = []
    totalTime = time()
    for i in range(kmeans.n_clusters):
        clusterIndices = np.where(kmeans.labels_ == i)
        clf = svm.SVC()
        clf.fit(reduced_data[clusterIndices], labels[clusterIndices])
        SVMModels.append(clf)
    print('SVM TRAIN TIME: ',str(time()-totalTime))
    svmOutput = np.append(svmOutput,time()-totalTime) #Store Training Time
    
    totalTime = time()
    testClusters = kmeans.predict(reduced_testdata)
    totalScore = 0
    for i in range(kmeans.n_clusters):
        t0 = time()
        clusterIndices = np.where(testClusters == i)
        score =SVMModels[i].score(reduced_testdata[clusterIndices],y_test[clusterIndices])
        totalScore = totalScore + score
    print('SVM SCORE TIME: ',str(time()-totalTime))
    svmOutput = np.append(svmOutput,time()-totalTime) #Store Score Time
    print("Overall Accuracy is :", str(totalScore/kmeans.n_clusters))
    svmOutput = np.append(svmOutput,(totalScore/kmeans.n_clusters)) #Store Accuracy
    if count_PC == 10:
        NonMNIST_svm_PerPC[0]=svmOutput
    else:
        NonMNIST_svm_PerPC= np.append(NonMNIST_svm_PerPC, [svmOutput], axis=0)

#STORE NonMNIST_svm_PerPC on DISK
pickle_file = os.path.join(".", 'NonMNIST_svm_PerPC.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'svm_PerPC': NonMNIST_svm_PerPC
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

NonMNIST_svm_PerPC= pickle.load( open( "NonMNIST_svm_PerPC.pickle", "rb" ) )
NonMNIST_svm_PerPC = NonMNIST_svm_PerPC['svm_PerPC']
NonMNIST_svm_PerPC = np.array(NonMNIST_svm_PerPC, dtype=object)  
#Plotly for Per PC Score
trace1 = go.Scatter(
    x=[str(round((i/784)*100,2))+'%' for i in range(10,110,10)],
    y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_svm_PerPC[0:11,3]],
    mode='lines+markers',
    name='PC Accuracy',
    hoverinfo='name',
    line=dict(
        shape='spline'
    )
)
data = [trace1]
layout = dict(
    yaxis=YAxis(title='Accuracy %'),
    xaxis=XAxis(title='% Data used after PCA'),
    title= 'SVM Accuracy By Data Used',
    )
fig = go.Figure(data=data, layout=layout)
#fig = dict(data=data, layout=layout)
py.iplot(fig, filename='PC-Accuracy-NONMNIST')


PCA_COUNT = 20
#LogisticRegression


# Instantiate
totalTime = time()
lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)
# Fit

#scores = cross_val_score(lg, NonMNIST_data, NonMNIST_test, cv=5)
lg.fit(X_train, y_train)
print('Total LR Training Time :', time()-totalTime)
totalTime = time()
# Predict
y_pred = lg.predict(X_test)
print("Time for LR Predict: ",time()-totalTime)
# Score
score =metrics.accuracy_score(y_test, y_pred)
#print("Overall Accuracy LogisticRegression: %0.2f (+/- %0.2f)" %(score.mean(),scores.std()*2))

#Complete
'''
Total LR Training Time : 344.9589865207672
Time for LR Predict:  0.024064064025878906
score :0.89000000000000001
'''

#Cross Validation
'''
Total LR Training Time : 1218.0152356624603
Time for LR Predict:  0.0
Overall Accuracy LogisticRegression: 0.92 (+/- 0.03)
'''


# KMEANS + LogisticRegression 
print('KMEANS + LogisticRegression With PCA :')
#reduced_data = PCA(n_components=PCA_COUNT).fit_transform(X_train)
kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=1)
kmeans.fit(X_train)
#reduced_testdata = PCA(n_components=PCA_COUNT).fit_transform(X_test)
# Get Data and Lables by Cluster
LRModels = []

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
tLRStartTime = time()

for i in range(kmeans.n_clusters):
	t0 = time()
	clusterIndices = np.where(kmeans.labels_ == i)
	lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)
    # Fit
	lg.fit(X_train[clusterIndices], labels[clusterIndices])
	LRModels.append(lg)

print('Total LR Training Time :', time()-tLRStartTime, 's')

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
print("Time for LR Predict: ",time()-tLRStartTime,'s')
print("Overall Accuracy LogisticRegression: ", str(totalScore/kmeans.n_clusters))

# LogisticRegression WITHOUT PCA

'''
WITHOUT PCA
Total LR Training Time : 287.13512325286865 s
Time for LR Predict:  0.0 s
Overall Accuracy LogisticRegression:  0.914466582536
'''

# LogisticRegression KMEANS WITH PCA

'''
KMEANS + LogisticRegression With PCA :
WITH PCA
Total LR Training Time : 13.095189094543457 s
Time for LR Predict:  0.0 s
Overall Accuracy LogisticRegression:  0.501344041384
'''


NonMNIST_KMeans_LR = np.array([['LogisticRegression Complete',344.9589865207672,0.89000],
             ['KMeans + LogisticRegression', 287.13512325286865, 0.914466582536],
             ['PCA + KMeans + LogisticRegression',13.095189094543457,0.501344041384]])

#PlotLy Bar PLOT
#Accuracy-by-LogisticRegression-Pipeline
trace = Bar(
        x= [name for name in NonMNIST_KMeans_LR[:,0]],
        y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_LR[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_LR[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy by LogisticRegression Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='NonMNIST-Accuracy-by-LogisticRegression-Pipeline')

#Computation-Time-by-LogisticRegression-Pipeline
trace = Bar(
        x= [name for name in NonMNIST_KMeans_LR[:,0]],
        y=[round(float(time),2) for time in NonMNIST_KMeans_LR[:,1]],
        text=[round(float(time),2) for time in NonMNIST_KMeans_LR[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time by LogisticRegression Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='NonMNIST-Computation-Time-by-LogisticRegression-Pipeline')

    
#SVM COMPLETE

totalTime = time()
clf = svm.SVC()
#clf.fit(data, y_train)  
scores = cross_val_score(clf, NonMNIST_data, NonMNIST_test, cv=5)
print("SVM Cross Validation Total Time :", (time()-totalTime))
#print("Overall Accuracy:",clf.score(X_test, y_test))
print("Overall Accuracy LogisticRegression: %0.2f (+/- %0.2f)" %(score.mean(),scores.std()*2))

'''
SVM Cross Validation Total Time : 4240.050569057465
Overall Accuracy LogisticRegression: 0.92 (+/- 0.02)
'''

# KMEANS + SVM 

reduced_data = PCA(n_components=PCA_COUNT).fit_transform(data)
kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=1)
kmeans.fit(reduced_data)
reduced_testdata = PCA(n_components=PCA_COUNT).fit_transform(X_test)
# Get Data and Lables by Cluster
SVMModels = []
totalTime = time()
for i in range(kmeans.n_clusters):
    clusterIndices = np.where(kmeans.labels_ == i)
    clf = svm.SVC()
    clf.fit(reduced_data[clusterIndices], labels[clusterIndices])
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


# SVM KMEANS WITHOUT PCA
'''
OUTPUT WITHOUT PCA
SVM TRAIN TIME:  201.9527885913849
SVM SCORE TIME:  16.640766382217407
Overall Accuracy is : 0.923599183548
'''

# SVM KMEANS WITH PCA
'''
WITH PCA
SVM TRAIN TIME:  10.07920789718628
SVM SCORE TIME:  0.6094698905944824
Overall Accuracy is : 0.539108300752

'''

NonMNIST_KMeans_SVM = np.array([['SVM Complete', float(4240.050569057465/5), 0.92],
             ['KMeans + SVM', 217.9527885913849, 0.923599183548],
             ['PCA + KMeans + SVM', 10.67920789718628, 0.539108300752]])

#PlotLy Bar PLOT
#Accuracy-by-SVM-Pipeline
trace = Bar(
        x= [name for name in NonMNIST_KMeans_SVM[:,0]],
        y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_SVM[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_SVM[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy by SVM Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='NonMNIST-Accuracy-by-SVM-Pipeline')

#Computation-Time-by-SVM-Pipeline
trace = Bar(
        x= [name for name in NonMNIST_KMeans_SVM[:,0]],
        y=[round(float(time),2) for time in NonMNIST_KMeans_SVM[:,1]],
        text=[round(float(time),2) for time in NonMNIST_KMeans_SVM[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time by SVM Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='NonMNIST-Computation-Time-by-SVM-Pipeline')


#DecisionTree COMPLETE
totalTime = time()
clf = tree.DecisionTreeClassifier()
#clf.fit(data, labels)
#print('DT TRAIN TIME: ',str(time()-totalTime))
#totalTime = time()
#y_predict =clf.predict(X_test)
#score =metrics.accuracy_score(y_test, y_predict)
scores = cross_val_score(clf, NonMNIST_data, NonMNIST_test, cv=5)
print('DT Cross Validation Total TIME: ',str(time()-totalTime))
#print("Overall Accuracy is :", str(score))
print("Overall Accuracy LogisticRegression: %0.2f (+/- %0.2f)" %(score.mean(),scores.std()*2))

'''
DT Cross Validation Total TIME:  100.86139965057373
Overall Accuracy LogisticRegression: 0.92 (+/- 0.03)
'''


#OUTPUT DT COMPLETE
'''
DT TRAIN TIME:  30.83156657218933
DT TRAIN TIME:  0.0
Overall Accuracy is : 0.8498
'''

# KMEANS + DecisionTree

#reduced_data = PCA(n_components=PCA_COUNT).fit_transform(X_train)
kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=1)
kmeans.fit(X_train)
#reduced_testdata = PCA(n_components=PCA_COUNT).fit_transform(X_test)
# Get Data and Lables by Cluster
DTModels = []
totalTime = time()
for i in range(kmeans.n_clusters):
    clusterIndices = np.where(kmeans.labels_ == i)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train[clusterIndices], y_train[clusterIndices])
    DTModels.append(clf)
print('DT TRAIN TIME: ',str(time()-totalTime))

totalTime = time()
from sklearn import metrics
testClusters = kmeans.predict(X_test)
totalScore = 0
for i in range(kmeans.n_clusters):
    t0 = time()
    clusterIndices = np.where(testClusters == i)
    y_predict =DTModels[i].predict(X_test[clusterIndices])
    score =metrics.accuracy_score(y_test[clusterIndices], y_predict)
    totalScore = totalScore + score
print('DT SCORE TIME: ',str(time()-totalTime))
print("Overall Accuracy is :", str(totalScore/kmeans.n_clusters))

#OUTPUT WITHOUT PCA
'''
DT TRAIN TIME:  15.923589944839478
DT SCORE TIME:  0.015596628189086914
Overall Accuracy is : 0.872814928443
'''

#OUTPUT WITH PCA
'''
DT TRAIN TIME:  1.5314126014709473
DT SCORE TIME:  0.0
Overall Accuracy is : 0.39974973622
'''

NonMNIST_KMeans_DT = np.array([['DT Complete', 30.83156657218933, 0.8498],
             ['KMeans + DT', 16.123589944839478, 0.872814928443],
             ['PCA + KMeans + DT', 1.5314126014709473, 0.3997497362]])


#PlotLy Bar PLOT
#Accuracy-by-DecisionTree-Pipeline
trace = Bar(
        x= [name for name in NonMNIST_KMeans_DT[:,0]],
        y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_DT[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_DT[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy by DecisionTree Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='NonMNIST-Accuracy-by-DecisionTree-Pipeline')

#Computation-Time-by-DecisionTree-Pipeline
trace = Bar(
        x= [name for name in NonMNIST_KMeans_DT[:,0]],
        y=[round(float(time),2) for time in NonMNIST_KMeans_DT[:,1]],
        text=[round(float(time),2) for time in NonMNIST_KMeans_DT[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time by DecisionTree Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='NonMNIST-Computation-Time-by-DecisionTree-Pipeline')


#KNeighborsClassifier Complete
totalTime = time()
kNC = KNeighborsClassifier(n_neighbors=3)
#kNC.fit(X_train,y_train)
#print('KNeighborsClassifier Complete TRAIN TIME: ',str(time()-totalTime))
totalTime = time()
#score =kNC.score(X_test,y_test)
#print('KNeighborsClassifier Complete SCORE TIME: ',str(time()-totalTime))
#print("KNeighborsClassifier Complete Accuracy is :", score)
scores = cross_val_score(kNC, NonMNIST_data, NonMNIST_test, cv=5)
print('KNeighborsClassifier Complete Cross Validation Total TIME: ',str(time()-totalTime))
#print("Overall Accuracy is :", str(score))
print("Overall Accuracy KNeighborsClassifier Complete: %0.2f (+/- %0.2f)" %(score.mean(),scores.std()*2))

#CrossValidation
'''
KNeighborsClassifier Complete Cross Validation Total TIME:  3199.9726734161377
Overall Accuracy KNeighborsClassifier Complete: 0.92 (+/- 0.03)
'''


#COMPLETE
'''
KNeighborsClassifier Complete TRAIN TIME:  7.161088943481445
KNeighborsClassifier Complete SCORE TIME:  295.05201959609985
KNeighborsClassifier Complete Accuracy is : 0.915
''' 

# KMEANS + KNeighborsClassifier 

#reduced_data = PCA(n_components=PCA_COUNT).fit_transform(X_train)
kmeans = KMeans(n_clusters=CLUSTER_COUNT, n_init=1)
kmeans.fit(X_train)
#reduced_testdata = PCA(n_components=PCA_COUNT).fit_transform(X_test)
# Get Data and Lables by Cluster
KNCModels = []
totalTime = time()
for i in range(kmeans.n_clusters):
    clusterIndices = np.where(kmeans.labels_ == i)
    kNC = KNeighborsClassifier(n_neighbors=3)
    kNC.fit(reduced_data[clusterIndices],y_train[clusterIndices])
    KNCModels.append(kNC)
print('kNC TRAIN TIME: ',str(time()-totalTime))

totalTime = time()

testClusters = kmeans.predict(reduced_testdata)
totalScore = 0
for i in range(kmeans.n_clusters):
    t0 = time()
    clusterIndices = np.where(testClusters == i)
    score =KNCModels[i].score(reduced_testdata[clusterIndices],y_test[clusterIndices])
    totalScore = totalScore + score
print('SVM SCORE TIME: ',str(time()-totalTime))
print("Overall Accuracy is :", str(totalScore/kmeans.n_clusters))

#OUTPUT WITHOUT PCA
'''
kNC TRAIN TIME:  2.7028279304504395
SVM SCORE TIME:  47.83090376853943
Overall Accuracy is : 0.902670789092
'''

#OUTPUT WITH PCA
'''
kNC TRAIN TIME:  0.11881613731384277
SVM SCORE TIME:  1.50612211227417
Overall Accuracy is : 0.522634337612
'''

NonMNIST_KMeans_kNC = np.array([['KNeighbors Complete', 302.161, 0.915],
             ['KMeans + KNeighbors', 50.7028279304504395, 0.902670789092],
             ['PCA + KMeans + KNeighbors', 1.60612211227417, 0.522634337612]])

#PlotLy Bar PLOT
#Accuracy-by-KNeighbors-Pipeline
trace = Bar(
        x= [name for name in NonMNIST_KMeans_kNC[:,0]],
        y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_kNC[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_kNC[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy by KNeighbors Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='NonMNIST-Accuracy-by-KNeighbors-Pipeline')

#Computation-Time-by-KNeighbors-Pipeline
trace = Bar(
        x= [name for name in NonMNIST_KMeans_kNC[:,0]],
        y=[round(float(time),2) for time in NonMNIST_KMeans_kNC[:,1]],
        text=[round(float(time),2) for time in NonMNIST_KMeans_kNC[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

data = Data([trace])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time by KNeighbors Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='NonMNIST-Computation-Time-by-KNeighbors-Pipeline')



#PlotLy Bar PLOT for all Accuracy
NonMNIST_KMeans_MNIST = np.column_stack((np.column_stack((np.column_stack((NonMNIST_KMeans_LR,NonMNIST_KMeans_SVM)),NonMNIST_KMeans_DT)),NonMNIST_KMeans_kNC))

trace1 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_LR[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_LR[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)),
        name="Logistic Regression")

trace2 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="SVM",
        y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_SVM[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_SVM[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))


trace3 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="Decision Tree",
        y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_DT[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_DT[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

trace4 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="KNeighbors Classifier",
        y=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_kNC[:,2]],
        text=[round(float(accuracy)*100,2) for accuracy in NonMNIST_KMeans_kNC[:,2]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))


data = Data([trace1,trace2,trace3,trace4])

layout=Layout(
        yaxis=YAxis(title='Accuracy in %'),
        title='Accuracy for NonMNIST Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Accuracy-for-NonMNIST-Pipeline')


#PlotLy Bar PLOT for all Computation-Time-
#KMeans_MNIST = np.column_stack((np.column_stack((np.column_stack((KMeans_LR,KMeans_SVM)),KMeans_DT)),KMeans_kNC))

trace1 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        y=[round(float(accuracy),2) for accuracy in NonMNIST_KMeans_LR[:,1]],
        text=[round(float(accuracy),2) for accuracy in NonMNIST_KMeans_LR[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)),
        name="Logistic Regression")

trace2 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="SVM",
        y=[round(float(accuracy),2) for accuracy in NonMNIST_KMeans_SVM[:,1]],
        text=[round(float(accuracy),2) for accuracy in NonMNIST_KMeans_SVM[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))


trace3 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="Decision Tree",
        y=[round(float(accuracy),2) for accuracy in NonMNIST_KMeans_DT[:,1]],
        text=[round(float(accuracy),2) for accuracy in NonMNIST_KMeans_DT[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))

trace4 = Bar(
        x= ['Complete','KMeans + Method','PCA + KMeans + Method'],
        name="KNeighbors Classifier",
        y=[round(float(accuracy),2) for accuracy in NonMNIST_KMeans_kNC[:,1]],
        text=[round(float(accuracy),2) for accuracy in NonMNIST_KMeans_kNC[:,1]],
        textposition = 'auto',
        marker=dict(line=dict(width=1.5)))


data = Data([trace1,trace2,trace3,trace4])

layout=Layout(
        yaxis=YAxis(title='Computation Time in Secs'),
        title='Computation Time for NonMNIST Pipeline')

mnist_Pipeline = Figure(data=data, layout=layout)
py.plot(mnist_Pipeline, filename='Computation-Time-by-NonMNIST-Pipeline')