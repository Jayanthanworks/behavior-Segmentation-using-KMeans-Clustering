# behavior-Segmentation-using-KMeans-Clustering
Customer Segmentation using K Means Clustering
In [189]:
# Business Background : 
# Industry :FMCD (Consumer Durable)
# The Retail Brand Store 'PSP associates' has got a database of 18678 unique customers from 2011 to 2016.
# The Sales data is of datewise/monthwise and model wise purchase

# Business Objective : 
#1) Behaviour segementation -To perform customer segmentation based on Purchase history
#2) To understand the retail consumer purchase pattern 
#3) To drive special loyalty program for Value customers  
#3) To Identify segments for Targeted marketing activities for the store
 
In [ ]:
#  Approach to Data Analysis: Goal - To find out optimum clusters for the dataset and label each customer id to cluster group

# 1) Machine learning- Unsupervised Algorithm - K means clustering 
# Steps : Start with EDA to understand the datapattern
#         Apply RFM analysis -(Recency Freuency Monetary) to individual customers
#         Encode with a Scale of 1 to 5 for the generated RFM paramter
#         Apply Elbow graph for finding the optimum cluster required
#         Apply K means alogorithm in the cleaned dataset
#         Come out with cluster formation for individual customer id
#         Data visualisation of formed clusters for RFM paramaters with respect to each other.
In [1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import seaborn as sns

os.chdir ("C:/Users/Jayanthan/Downloads/POC")
data = pd.read_excel('POC in python.xlsx')
In [2]:
data.describe()
Out[2]:

Invoice No.
Qty
Invoice Value
Customer ID
count
21076.000000
21076.000000
21076.000000
2.107600e+04
mean
10739.500000
1.018599
21159.309831
8.188147e+09
std
6084.261473
0.306103
16945.645504
5.619476e+09
min
202.000000
1.000000
300.000000
4.243560e+08
25%
5470.750000
1.000000
12000.000000
8.608872e+09
50%
10739.500000
1.000000
15500.000000
9.597353e+09
75%
16008.250000
1.000000
25900.000000
9.843556e+09
max
21277.000000
29.000000
430000.000000
9.999479e+10
In [3]:
data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 21076 entries, 0 to 21075
Data columns (total 8 columns):
Date                         21076 non-null datetime64[ns]
Customer/Dealer/Financier    14474 non-null object
Invoice No.                  21076 non-null int64
Item/Model                   21076 non-null object
Qty                          21076 non-null int64
Invoice Value                21076 non-null int64
Customer Name                21076 non-null object
Customer ID                  21076 non-null int64
dtypes: datetime64[ns](1), int64(4), object(3)
memory usage: 1.3+ MB
In [3]:
data.shape
Out[3]:
(21076, 8)
In [4]:
data.nunique()
Out[4]:
Date                          1616
Customer/Dealer/Financier     6250
Invoice No.                  21076
Item/Model                    1555
Qty                             10
Invoice Value                  803
Customer Name                13052
Customer ID                  18676
dtype: int64
In [5]:
data.isna().any()
Out[5]:
Date                         False
Customer/Dealer/Financier     True
Invoice No.                  False
Item/Model                   False
Qty                          False
Invoice Value                False
Customer Name                False
Customer ID                  False
dtype: bool
In [8]:
plt.hist(data['Invoice Value'],8)
Out[8]:
(array([1.973e+04, 1.283e+03, 4.300e+01, 1.400e+01, 4.000e+00, 1.000e+00,
        0.000e+00, 1.000e+00]),
 array([3.000000e+02, 5.401250e+04, 1.077250e+05, 1.614375e+05,
        2.151500e+05, 2.688625e+05, 3.225750e+05, 3.762875e+05,
        4.300000e+05]),
 <a list of 8 Patch objects>)

 
In [9]:
plt.hist(data['Qty'],8)
Out[9]:
(array([2.1067e+04, 4.0000e+00, 1.0000e+00, 0.0000e+00, 3.0000e+00,
        0.0000e+00, 0.0000e+00, 1.0000e+00]),
 array([ 1. ,  4.5,  8. , 11.5, 15. , 18.5, 22. , 25.5, 29. ]),
 <a list of 8 Patch objects>)

 
In [195]:
sns.distplot(data['Customer ID'])

C:\Users\Jayanthan\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
  warnings.warn("The 'normed' kwarg is deprecated, and has been "
Out[195]:
<matplotlib.axes._subplots.AxesSubplot at 0x16d2d1fed30>

 
In [198]:
sns.distplot(data['Invoice Value'],10)

C:\Users\Jayanthan\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
  warnings.warn("The 'normed' kwarg is deprecated, and has been "
Out[198]:
<matplotlib.axes._subplots.AxesSubplot at 0x16d2ea70208>

 
In [12]:
data_num=data.select_dtypes(include=['float64','int64'])
data_num.shape
Out[12]:
(21076, 4)
In [13]:
for i in range(0,len(data.columns),7):
    sns.pairplot(data_num,y_vars=['Customer ID'],x_vars=data_num.columns[0:7])

 

 
In [14]:
# No correlation between data
data.corr = data_num.corr()['Customer ID']
data.corr
Out[14]:
Invoice No.     -0.049113
Qty             -0.004895
Invoice Value   -0.090538
Customer ID      1.000000
Name: Customer ID, dtype: float64
In [ ]:
corr_bf= bfcer_num.drop('cups',axis=1).corr()
sns.heatmap(corr_bf[(corr_bf>=0.5)|(corr_bf<=-0.4)],
            cmap='viridis',vmax=1.0,vmin=-1.0,linewidths=0.1,
            annot=True,annot_kws={"size":8},square=True);
In [168]:
plt.boxplot(data['Invoice Value'])
Out[168]:
{'whiskers': [<matplotlib.lines.Line2D at 0x16d2ed37ac8>,
  <matplotlib.lines.Line2D at 0x16d2ed37f60>],
 'caps': [<matplotlib.lines.Line2D at 0x16d2ed483c8>,
  <matplotlib.lines.Line2D at 0x16d2ed487f0>],
 'boxes': [<matplotlib.lines.Line2D at 0x16d2ed37978>],
 'medians': [<matplotlib.lines.Line2D at 0x16d2ed48c18>],
 'fliers': [<matplotlib.lines.Line2D at 0x16d2ed5e080>],
 'means': []}

 
In [7]:
import datetime as dt

print('Most recent invoice is from:')
print(data['Date'].max())

lastDate = dt.datetime(2016,12,4)
data['Date'] = pd.to_datetime(data['Date'])

Most recent invoice is from:
2016-12-03 00:00:00
In [26]:
rfmTable = data.groupby('Customer ID').agg({'Date': lambda x: (lastDate - x.max()).days, 
                                           'Invoice No.': lambda x: len(x), 
                                           'Invoice Value': lambda x: x.sum()})
rfmTable['Date'] = rfmTable['Date'].astype(int)
rfmTable.rename(columns={'Date': 'recency', 
                         'Invoice No.': 'frequency', 
                         'Invoice Value': 'monetary'}, inplace=True)

rfmTable.head()
Out[26]:

recency
frequency
monetary
Customer ID



424355975
256
2
70000
1010010012
2437
1
11900
1010010017
2437
1
2750
1010010020
2432
1
2800
1010010021
2432
1
16699
In [111]:
f_score = []
m_score = []
r_score = []

columns = ['frequency', 'monetary']
scores_str = ['f_score', 'm_score']
scores = [f_score, m_score]
for n in range(len(columns)):
    rfmTable = rfmTable.sort_values(columns[n], ascending=False)
    
In [112]:
refs = np.arange(1,18679)
rfmTable['refs'] = refs
for i, row in rfmTable.iterrows():
        if row['refs'] <= 3730:
            scores[n].append(5)
        elif row['refs'] > 3730 and row['refs'] <= 3730*2:
            scores[n].append(4)
        elif row['refs'] > 3730*2 and row['refs'] <= 3730*3:
            scores[n].append(3)
        elif row['refs'] > 3730*3 and row['refs'] <= 3730*4:
            scores[n].append(2)
        else: 
            scores[n].append(1)
In [114]:
rfmTable = rfmTable.sort_values('frequency', ascending=False)

refs = np.arange(1,18679)
rfmTable['refs'] = refs
for i, row in rfmTable.iterrows():
        if row['refs'] <= 3730:
            f_score.append(5)
        elif row['refs'] > 3730 and row['refs'] <= 3730*2:
            f_score.append(4)
        elif row['refs'] > 3730*2 and row['refs'] <= 3730*3:
            f_score.append(3)
        elif row['refs'] > 3730*3 and row['refs'] <= 3730*4:
            f_score.append(2)
        else: 
            f_score.append(1)
In [115]:
rfmTable['f_score'] = f_score
In [116]:
# For recency, we do the opposite: most recents are better, so we order as ascending
rfmTable = rfmTable.sort_values('recency', ascending=True)
    
# Recreate index
refs = np.arange(1,18679)
rfmTable['refs'] = refs
# Add score
for i, row in rfmTable.iterrows():
    if row['refs'] <= 3730:
        r_score.append(5)
    elif row['refs'] > 3730 and row['refs'] <= 3730*2:
        r_score.append(4)
    elif row['refs'] > 3730*2 and row['refs'] <= 3730*3:
        r_score.append(3)
    elif row['refs'] > 3730*3 and row['refs'] <= 3730*4:
        r_score.append(2)
    else: 
        r_score.append(1)
In [117]:
# Create f_score column
rfmTable['r_score'] = r_score
    
rfmTableScores = rfmTable.drop(['frequency', 'monetary', 'recency', 'refs'], axis=1)
rfmTableScores.head(5)
Out[117]:

m_score
r_score
f_score
Customer ID



9965298739
2
5
4
9751030878
5
5
5
9842260649
5
5
2
9003508705
2
5
5
1010034702
5
5
2
In [124]:
from sklearn.cluster import KMeans    

wcss = []       #within-cluster sums of squares 
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0) # k-means - Initial cluster centers
    kmeans.fit(rfmTableScores)
    wcss.append(kmeans.inertia_)  # Sum of squared distance
    
In [128]:
plt.plot(range(1,11), wcss)
plt.title('Elbow graph')
plt.xlabel('Cluster number')
plt.ylabel('WCSS')
plt.show()

 
In [133]:
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
clusters = kmeans.fit_predict(rfmTableScores)
rfmTable['clusters'] = clusters
rfmTable.head()
Out[133]:

recency
frequency
monetary
refs
m_score
r_score
f_score
clusters
Customer ID








9965298739
1
1
13600
1
2
5
4
1
9751030878
1
2
40000
2
5
5
5
2
9842260649
1
1
62000
3
5
5
2
4
9003508705
1
1
12500
4
2
5
5
1
1010034702
1
1
37000
5
5
5
2
4
In [134]:
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(rfmTable.recency, rfmTable.frequency, rfmTable.monetary, s=50)

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
Out[134]:
Text(0.5,0,'Monetary')

 
In [139]:
fig = plt.figure(figsize=(15,10))
dx = fig.add_subplot(111, projection='3d')
colors = ['blue', 'yellow', 'green', 'red','black']

for i in range(0,5):
    dx.scatter(rfmTable[rfmTable.clusters == i].recency, 
               rfmTable[rfmTable.clusters == i].frequency, 
               rfmTable[rfmTable.clusters == i].monetary, 
               c = colors[i], 
               label = 'Cluster ' + str(i+1), 
               s=50)

dx.set_title('Clusters of customers')
dx.set_xlabel('Recency')
dx.set_ylabel('Frequency')
dx.set_zlabel('Monetary')
dx.legend()
Out[139]:
<matplotlib.legend.Legend at 0x16d2c9dcc88>

 
In [140]:
c1 = rfmTable[rfmTable.clusters == 0]
c2 = rfmTable[rfmTable.clusters == 1]
c3 = rfmTable[rfmTable.clusters == 2]
c4 = rfmTable[rfmTable.clusters == 3]
c5 = rfmTable[rfmTable.clusters == 4]

plt.scatter(c1.recency, c1.frequency, c = 'blue', label = 'Cluster 1')
plt.scatter(c2.recency, c2.frequency, c = 'yellow', label = 'Cluster 2')
plt.scatter(c3.recency, c3.frequency, c = 'green', label = 'Cluster 3')
plt.scatter(c4.recency, c4.frequency, c = 'red', label = 'Cluster 4')
plt.scatter(c5.recency, c5.frequency, c = 'black', label = 'Cluster 5')

plt.title('Clusters of customers')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.legend()
Out[140]:
<matplotlib.legend.Legend at 0x16d2c190f98>

 
In [141]:
c1 = rfmTable[rfmTable.clusters == 0]
c2 = rfmTable[rfmTable.clusters == 1]
c3 = rfmTable[rfmTable.clusters == 2]
c4 = rfmTable[rfmTable.clusters == 3]
c5 = rfmTable[rfmTable.clusters == 4]

plt.scatter(c1.frequency, c1.monetary, c = 'blue', label = 'Cluster 1')
plt.scatter(c2.frequency, c2.monetary, c = 'yellow', label = 'Cluster 2')
plt.scatter(c3.frequency, c3.monetary, c = 'green', label = 'Cluster 3')
plt.scatter(c4.frequency, c4.monetary, c = 'red', label = 'Cluster 4')
plt.scatter(c5.recency, c5.frequency, c = 'black', label = 'Cluster 5')

plt.title('Clusters of customers')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.legend()
Out[141]:
<matplotlib.legend.Legend at 0x16d2cdb5ef0>

 
In [142]:
c1 = rfmTable[rfmTable.clusters == 0]
c2 = rfmTable[rfmTable.clusters == 1]
c3 = rfmTable[rfmTable.clusters == 2]
c4 = rfmTable[rfmTable.clusters == 3]
c5 = rfmTable[rfmTable.clusters == 4]

plt.scatter(c1.recency, c1.monetary, c = 'blue', label = 'Cluster 1')
plt.scatter(c2.recency, c2.monetary, c = 'yellow', label = 'Cluster 2')
plt.scatter(c3.recency, c3.monetary, c = 'green', label = 'Cluster 3')
plt.scatter(c4.recency, c4.monetary, c = 'red', label = 'Cluster 4')
plt.scatter(c5.recency, c5.frequency, c = 'black', label = 'Cluster 5')

plt.title('Clusters of clients')
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.legend()
Out[142]:
<matplotlib.legend.Legend at 0x16d2ce21dd8>

 
