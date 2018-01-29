
# Analyzing and Predicting Facebook Post Engagement

### Project Structure
1. Project Question and Dataset Description
2. Loading and viewing dataset
3. Pre-processing
4. Exploratory Data Analysis
5. Modeling 
    - Outlier removal and re-shaping dataframe
    - Linear Regression
    - Random Forest Regression
6. Take-aways
7. References

Data comes via the UCI dataset: https://archive.ics.uci.edu/ml/datasets/Facebook+metrics

## 1. Question: Which posts have the highest engagement, and can we predict which posts will get higher engagement?
#### Assumptions:
- higher Facebook engagement leads to more downloads and overall engagement with the app
- There have been no changes in FB's feed algorithm since the data was posted (it has been already)

#### Dataset info
##### Variables
- Page total likes: how many likes the page had when post went live
- Type: what kind of post (video, link, status, photo)
- Category: not sure, 3 levels
- Post Month
- Post Weekday
- Post Hour
- Paid
- like
- share
- comment
- total interactions
- Post metrics (recorded after posting, EDA only)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## 2. Loading and Viewing the data


```python
df = pd.read_csv("~/Desktop/Facebook_metrics/dataset_Facebook.csv",delimiter=';')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 19 columns):
    Page total likes                                                       500 non-null int64
    Type                                                                   500 non-null object
    Category                                                               500 non-null int64
    Post Month                                                             500 non-null int64
    Post Weekday                                                           500 non-null int64
    Post Hour                                                              500 non-null int64
    Paid                                                                   499 non-null float64
    Lifetime Post Total Reach                                              500 non-null int64
    Lifetime Post Total Impressions                                        500 non-null int64
    Lifetime Engaged Users                                                 500 non-null int64
    Lifetime Post Consumers                                                500 non-null int64
    Lifetime Post Consumptions                                             500 non-null int64
    Lifetime Post Impressions by people who have liked your Page           500 non-null int64
    Lifetime Post reach by people who like your Page                       500 non-null int64
    Lifetime People who have liked your Page and engaged with your post    500 non-null int64
    comment                                                                500 non-null int64
    like                                                                   499 non-null float64
    share                                                                  496 non-null float64
    Total Interactions                                                     500 non-null int64
    dtypes: float64(3), int64(15), object(1)
    memory usage: 74.3+ KB



```python
df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Page total likes</th>
      <th>Type</th>
      <th>Category</th>
      <th>Post Month</th>
      <th>Post Weekday</th>
      <th>Post Hour</th>
      <th>Paid</th>
      <th>Lifetime Post Total Reach</th>
      <th>Lifetime Post Total Impressions</th>
      <th>Lifetime Engaged Users</th>
      <th>Lifetime Post Consumers</th>
      <th>Lifetime Post Consumptions</th>
      <th>Lifetime Post Impressions by people who have liked your Page</th>
      <th>Lifetime Post reach by people who like your Page</th>
      <th>Lifetime People who have liked your Page and engaged with your post</th>
      <th>comment</th>
      <th>like</th>
      <th>share</th>
      <th>Total Interactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>139441</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>2752</td>
      <td>5091</td>
      <td>178</td>
      <td>109</td>
      <td>159</td>
      <td>3078</td>
      <td>1640</td>
      <td>119</td>
      <td>4</td>
      <td>79.0</td>
      <td>17.0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139441</td>
      <td>Status</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>10</td>
      <td>0.0</td>
      <td>10460</td>
      <td>19057</td>
      <td>1457</td>
      <td>1361</td>
      <td>1674</td>
      <td>11710</td>
      <td>6112</td>
      <td>1108</td>
      <td>5</td>
      <td>130.0</td>
      <td>29.0</td>
      <td>164</td>
    </tr>
    <tr>
      <th>2</th>
      <td>139441</td>
      <td>Photo</td>
      <td>3</td>
      <td>12</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>2413</td>
      <td>4373</td>
      <td>177</td>
      <td>113</td>
      <td>154</td>
      <td>2812</td>
      <td>1503</td>
      <td>132</td>
      <td>0</td>
      <td>66.0</td>
      <td>14.0</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



Because columns 7 to 15 ("Lifetime post total impressions","Lifetime Engaged users", etc.) are recorded after posting, they will not be used for modeling. However, they can give useful information about post reach, and we will be looking at them in EDA.


```python

#df.drop(df.columns[7:15],
#        axis=1,inplace=True)
```

We need to fill the values that are blank with 0:

Goal: predict Total Interactions based on features

I have excluded comment, like, and share and instead focused on Total Interactions, which is what we will be modeling for. 

An outlier is easily visible for total interactions, at around 6000. 

### Pre-processing


```python
df['like'].describe()
```




    count     499.000000
    mean      177.945892
    std       323.398742
    min         0.000000
    25%        56.500000
    50%       101.000000
    75%       187.500000
    max      5172.000000
    Name: like, dtype: float64




```python
df['Type'].value_counts()
```




    Photo     426
    Status     45
    Link       22
    Video       7
    Name: Type, dtype: int64




```python
df['Paid'].value_counts()
df['like'].fillna(0,inplace=True)
df['share'].fillna(0,inplace=True)
df['Paid'].fillna(0,inplace=True)
```


```python
#df.drop(df.columns[1],axis=1,inplace = True)
```

## EDA


```python
df['Type'] = df['Type'].apply(lambda x: str(x))
plotdf = df.drop(df.columns[7:15],axis =1)
sns.pairplot(data=plotdf)
```




    <seaborn.axisgrid.PairGrid at 0x14ad25050>




![png](output_19_1.png)



```python
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),cmap='viridis',annot=True,cbar=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17bba0f90>




![png](output_20_1.png)


#### Univariate- Page Likes and Post Likes


```python
sns.distplot(df['Page total likes'],bins=20)
plt.title("Distribution of Total Page Likes")
```




    <matplotlib.text.Text at 0x169333f10>




![png](output_22_1.png)



```python
plt.figure(figsize=(14,8))
sns.distplot(df[df['like']<1000]['like'],bins=20,color='dodgerblue',hist_kws={'alpha':0.6})
plt.xlim(0,1000)
plt.xlabel("# of Likes",fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Distribution of Like/Post',fontsize=20)
#plt.savefig('NumLikes.png', bbox_inches='tight')
```




    <matplotlib.text.Text at 0x1770f2450>




![png](output_23_1.png)


#### Analysis of TIme Categories


```python
sns.countplot(df['Category'],palette='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17c0b6f10>




![png](output_25_1.png)



```python
sns.countplot(df['Post Month'],palette='viridis')
#plt.savefig('monthCount.png', bbox_inches='tight')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x175cb7cd0>




![png](output_26_1.png)



```python
from collections import OrderedDict
sns.set_style("whitegrid")

f, ax = plt.subplots()
sns.countplot(df['Post Weekday'],palette='viridis')
handles = ["Su", "M", "Tu", "W", "Th", "F",'Sa']
labels = [0,1,2,3,4,5,6]
plt.xticks(labels, handles)
ax.set_ylabel("Frequency")
sns.despine(offset = 5, trim = True)
plt.title("Frequency of Posts by Weekday")
#plt.savefig('dayCount.png', bbox_inches='tight')
```




    <matplotlib.text.Text at 0x16d6b99d0>




![png](output_27_1.png)



```python
sns.set_style("whitegrid")

f, ax = plt.subplots()
sns.countplot(df['Post Hour'],palette='viridis',)
#handles = ["Su", "M", "Tu", "W", "Th", "F",'Sa']
#labels = [0,1,2,3,4,5,6]
#plt.xticks(labels, handles)
ax.set_ylabel("Frequency")
sns.despine(offset = 5, trim = True)
plt.title("Frequency of Posts by Hour")
#plt.savefig('hourCount.png', bbox_inches='tight')
```




    <matplotlib.text.Text at 0x14aee7fd0>




![png](output_28_1.png)



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='Post Weekday',y='like',data=df,palette='viridis')
plt.ylim(0,750)
plt.title("Distribution of Likes/Post by Weekday Posted")
plt.ylabel("# of Likes")
#plt.savefig('dayBox.png', bbox_inches='tight')
```




    <matplotlib.text.Text at 0x16b00ee50>




![png](output_29_1.png)



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='Post Weekday',y='like',hue='Paid',data=df,palette='viridis')
plt.ylim(0,2200)
plt.title("Distribution of Post Likes by Weekday Posted")
plt.ylabel("# of Likes")

```




    <matplotlib.text.Text at 0x131562290>




![png](output_30_1.png)



```python
plt.figure(figsize=(8,6))
sns.countplot(x='Post Weekday',hue='Paid',data=df,palette='viridis')
plt.title("Distribution of Total Interactions by Weekday Posted")
```




    <matplotlib.text.Text at 0x12e0c2f90>




![png](output_31_1.png)


Facebook uses their own prestoDB, where the Day of the week (1-7), where Sunday is the first day of the week.



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='Post Hour',y='like',data=df,palette='viridis')
plt.ylim(0,750)
plt.title("Distribution of Post Likes by Hour Posted")
plt.ylabel("# of Likes")
plt.savefig('hourBox.png', bbox_inches='tight')
```


![png](output_33_0.png)



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='Post Hour',y='like',hue='Paid',data=df,palette='viridis')
plt.ylim(0,1750)
plt.title("Distribution of Total Interactions by Hour Posted")
plt.ylabel("# of Likes")
```




    <matplotlib.text.Text at 0x130e8e990>




![png](output_34_1.png)



```python
plt.figure(figsize=(8,6))
sns.countplot(x='Post Hour',hue='Paid',data=df,palette='viridis')
#plt.ylim(0,2200)
plt.title("Distribution of Total Interactions by Hour Posted")
```




    <matplotlib.text.Text at 0x12af34850>




![png](output_35_1.png)



```python
plt.figure(figsize=(8,6))
sns.stripplot(x='Post Hour',y='like',data=df,palette='viridis')
plt.ylim(0,2200)
plt.title("Distribution of Total Interactions by Hour Posted")
plt.ylabel("# of Likes")
```




    <matplotlib.text.Text at 0x12fe58290>




![png](output_36_1.png)



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='Post Month',y='like',data=df,palette='viridis')
plt.ylim(0,750)
plt.title("Distribution of Total Interactions by Month Posted")
plt.ylabel("# of Likes")
plt.savefig('monthBox.png', bbox_inches='tight')
```


![png](output_37_0.png)


When did the majority of posts go live?


```python
plt.figure(figsize=(8,6))
sns.boxplot(x='Post Month',y='like',hue='Paid',data=df,palette='viridis')
plt.ylim(0,2200)
plt.title("Distribution of Total Interactions by Month Posted")
plt.ylabel("# of Likes")
```




    <matplotlib.text.Text at 0x136070110>




![png](output_39_1.png)


#### Total Page Likes vs. Post likes


```python
sns.lmplot(x='Page total likes',y='like',
           hue='Paid',data=df,fit_reg=True,
          scatter_kws={'alpha':0.6},size=(6))
plt.legend()
plt.ylim(0,1000)
plt.ylabel("# of Likes")
plt.title('# Page Likes vs. # Post Likes')
#plt.savefig('pageLikeScatter.png', bbox_inches='tight')
```




    <matplotlib.text.Text at 0x1697a1750>




![png](output_41_1.png)


Does the number of likes per post increase as the total page like increases? It depends. There is a weak positive trend for paid posts but no increase in non-paid posts.

#### Category vs. Likes


```python
plt.figure(figsize=(8,6))
sns.countplot(x='Category',data=df,palette='viridis')
plt.ylabel("# of Likes")
plt.title("Number of Likes vs. Post Category")
plt.savefig('catCount.png', bbox_inches='tight')
```


![png](output_44_0.png)



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='Category',y='like',data=df,palette='viridis')
plt.ylim(0,750)
plt.ylabel("# of Likes")
plt.title("Number of Likes vs. Post Category")
plt.savefig('catBox.png', bbox_inches='tight')
```


![png](output_45_0.png)



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='Category',y='like',data=df,hue='Paid',palette='viridis')
plt.ylim(0,750)
plt.ylabel("# of likes")
plt.legend(loc='upper left')
plt.title("Likes/Post by Category and if Posts were Paid")
plt.savefig('catPaidCount.png', bbox_inches='tight')

```


![png](output_46_0.png)


Observations:

- Categories 2 and 3 peformed similarly, while Category 1 performed the worst
- The most posts were of category 1 posts
- Paid Category 1 posts saw the largest increase, followed by 2 and then 3. 


#### Post Type vs. Likes


```python
plt.figure(figsize=(10,6))
sns.boxplot(x='Type',y='like',data=df,palette='viridis')
plt.ylim(0,750)
#sns.despine(offset=4,bottom=True)
plt.title("Likes/Post by Post Type")
plt.ylabel("# of likes")
#plt.legend(loc='upper left')
plt.savefig('typeBox.png', bbox_inches='tight')

```


![png](output_49_0.png)



```python
plt.figure(figsize=(10,6))
sns.countplot(x='Type',data=df,palette='viridis')
#plt.ylim(0,750)
#sns.despine(offset=4,bottom=True)
plt.title("Number of Posts by Type")
plt.ylabel("# of likes")
#plt.legend(loc='upper left')
plt.savefig('typeCount.png', bbox_inches='tight')

```


![png](output_50_0.png)



```python
plt.figure(figsize=(10,6))
sns.boxplot(x='Type',y='like',hue='Paid',data=df,palette='viridis')
plt.ylim(0,750)
#sns.despine(offset=4,bottom=True)
plt.title("Total Interactions by Post Type")
plt.ylabel("# of likes")
plt.savefig('typePaidBox.png', bbox_inches='tight')
```


![png](output_51_0.png)



```python
plt.figure(figsize=(10,6))
sns.countplot(x='Type',hue='Paid',data=df,palette='viridis')
plt.ylim(0,320)
#sns.despine(offset=4,bottom=True)
plt.title("Frequency of Post Type by Paid")
plt.ylabel("Frequency")
plt.savefig('typePaidCount.png', bbox_inches='tight')
```


![png](output_52_0.png)


Video posts had the highest mean, median, and percentiles. Photo posts had the largest range, from 0 to 6000.

Take-aways:

- On average video posts had higher engagement
- Photo posts had the largest range
- This suggests that total interactions can depend on the photo posted
- Links performed the worst, with the lowest mean, range, and median
- Status on average performed 2nd best in terms on mean engagement, with the second largest range. 
- No difference in paid link posts
- Paid Status Posts had higher likes on average when compared to non-paid status posts
    - Possibly due to how well status updates integrate in newsfeed
- Paid Photo Posts performed marginally better than non-paid posts on average, though non-paid posts had a larger range (more outliers)

#### Paid Posts vs. Likes


```python
plt.figure(figsize=(10,6))
sns.boxplot(x='Paid',y='like',data=df,palette='viridis')
plt.ylim(0,500)
#sns.despine(offset=4,bottom=True)
plt.title("likes/Post by Paid/Non-Paid Posts")
plt.ylabel("# of likes")
plt.savefig('paidBox.png', bbox_inches='tight')
```


![png](output_55_0.png)



```python
from numpy import median
print(median(df[df['Paid']==0]['like']))
print(median(df[df['Paid']==1]['like']))
```

    96.0
    128.0



```python
plt.figure(figsize=(10,6))
sns.countplot(x='Paid',data=df,palette='viridis')
#sns.despine(offset=4,bottom=True)
plt.title("likes/Post by Paid/Non-Paid Posts")
plt.ylabel("# of Posts")
plt.savefig('paidCount.png', bbox_inches='tight')
```


![png](output_57_0.png)


Observations:
- There were more non-paid posts than paid posts
- Paid posts performed 30 likes by median better than non-paid posts

#### Which days and hours performed the best?


```python
df.columns
```




    Index([u'Page total likes', u'Type', u'Category', u'Post Month',
           u'Post Weekday', u'Post Hour', u'Paid', u'comment', u'like', u'share',
           u'Total Interactions', u'Video', u'Status', u'Photo', u'Cat_1',
           u'Cat_2'],
          dtype='object')




```python
timePivot = pd.pivot_table(df,aggfunc='median',
                           columns='Post Hour',
                           index='Post Weekday',
                           values='like')
timePivot = timePivot[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
```


```python
plt.figure(figsize=(18,10))
cmap= sns.cubehelix_palette(8, start=.5, rot=-.75,as_cmap=True)
sns.heatmap(timePivot,cbar=False,cmap=cmap,annot=True, fmt='g')
#plt.pcolor(lnch_pivot,cmap=plt.cm.Blues, alpha=0.8)
plt.yticks(np.arange(7),['Sa','F','Th','W','Tu','M','Su'],fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('Post Weekay',fontsize=20)
plt.xlabel('Post Hour',fontsize=20)
plt.title('Median Likes Per Post by Post Weekday and Hour',fontsize=20)
plt.savefig('medianLikeHeatmap.png', bbox_inches='tight')
```


![png](output_62_0.png)


# Modeling: Can we predict the number of likes a post will get?


```python
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
```

### Removing Outliers and Binarizing Categorical Variables

To remove the decent amount of outliers, I will remove nay variables that are above the 90th percentile.

After, I will span the categorical columns out into binary features. To avoid multicolinearity, I will be taking n-1 columns for each feature.

#### Removing Outliers


```python
outlierCut = np.percentile(df['like'],90)
outlierCut
```




    330.10000000000002




```python
df = df[df['like']<outlierCut]
```


```python
scaler = StandardScaler()
scaler.fit(df['Page total likes'])
#print(scaler.mean_)
df['Page total likes'] = scaler.transform(df['Page total likes'])
```

    //anaconda/lib/python2.7/site-packages/sklearn/preprocessing/data.py:586: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    //anaconda/lib/python2.7/site-packages/sklearn/preprocessing/data.py:649: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)


The function below will translate the weekdays to their labels, rather than 1-7.


```python
def Weekday(x):
    if x == 1:
        return 'Su'
    elif x== 2:
        return 'Mo'
    elif x == 3:
        return 'Tu'
    elif x == 4:
        return 'We'
    elif x == 5:
        return 'Th'
    elif x ==6:
        return 'Fr'
    elif x == 7:
        return "Sa"

df['Weekday'] = df['Post Weekday'].apply(lambda x: Weekday(x))
```

#### Spanning out categorical columns


```python
dayDf = pd.get_dummies(df['Weekday'])
```


```python
df = pd.concat([df,dayDf],axis=1)
```


```python
hours = list(range(0,18))
#hours
for i in hours:
    hours[i] = str(hours[i])
    hours[i]='hr_'+ hours[i]
    #print str(hours[i])
```


```python
hourDf = pd.get_dummies(df['Post Hour'],prefix='hr_')
df = pd.concat([df,hourDf],axis=1)
monthDf = pd.get_dummies(df['Post Month'],prefix='Mo')
df = pd.concat([df,monthDf],axis=1)
df['Video'] = pd.get_dummies(df['Type'])['Video']
df['Status'] = pd.get_dummies(df['Type'])['Status']
df['Photo'] = pd.get_dummies(df['Type'])['Photo']
df['Cat_1'] = pd.get_dummies(df['Category'])[1]
df['Cat_2'] = pd.get_dummies(df['Category'])[2]
#To avoid multicollinearity with the post types I am not including Links.
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Page total likes</th>
      <th>Type</th>
      <th>Category</th>
      <th>Post Month</th>
      <th>Post Weekday</th>
      <th>Post Hour</th>
      <th>Paid</th>
      <th>Lifetime Post Total Reach</th>
      <th>Lifetime Post Total Impressions</th>
      <th>Lifetime Engaged Users</th>
      <th>...</th>
      <th>Mo_8</th>
      <th>Mo_9</th>
      <th>Mo_10</th>
      <th>Mo_11</th>
      <th>Mo_12</th>
      <th>Video</th>
      <th>Status</th>
      <th>Photo</th>
      <th>Cat_1</th>
      <th>Cat_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.012339</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>2752</td>
      <td>5091</td>
      <td>178</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.012339</td>
      <td>Status</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>10</td>
      <td>0.0</td>
      <td>10460</td>
      <td>19057</td>
      <td>1457</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.012339</td>
      <td>Photo</td>
      <td>3</td>
      <td>12</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>2413</td>
      <td>4373</td>
      <td>177</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.012339</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>7244</td>
      <td>13594</td>
      <td>671</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.012339</td>
      <td>Status</td>
      <td>2</td>
      <td>12</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
      <td>10472</td>
      <td>20849</td>
      <td>1191</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>



### Train Test Split


```python
#
x = df[['Page total likes','Paid','Video','Status','Photo',
    'Cat_1','Cat_2','Mo','Tu','Sa',"We",'Th','Fr',
       'hr__17','hr__1','hr__2','hr__3','hr__4','hr__5', 'hr__6','hr__7','hr__8',
        'hr__9','hr__10','hr__11','hr__12','hr__13','hr__14','hr__15','hr__16','Mo_1',
       'Mo_2','Mo_12','Mo_4','Mo_5','Mo_6','Mo_7','Mo_8','Mo_9','Mo_11','Mo_10']]
y = df['like']
```

The train test split will be with a test size of 0.3, and a random state of 50. With 500 samples a 30% split should not lead to overfitting.


```python
x_train,x_test,y_train, y_test = train_test_split(x,
                                                  y, test_size=0.3,
                                                  random_state=50)
```


```python
y_test.count()
```




    135




```python
x_test.columns
```




    Index([u'Page total likes', u'Paid', u'Video', u'Status', u'Photo', u'Cat_1',
           u'Cat_2', u'Mo', u'Tu', u'Sa', u'We', u'Th', u'Fr', u'hr__17', u'hr__1',
           u'hr__2', u'hr__3', u'hr__4', u'hr__5', u'hr__6', u'hr__7', u'hr__8',
           u'hr__9', u'hr__10', u'hr__11', u'hr__12', u'hr__13', u'hr__14',
           u'hr__15', u'hr__16', u'Mo_1', u'Mo_2', u'Mo_12', u'Mo_4', u'Mo_5',
           u'Mo_6', u'Mo_7', u'Mo_8', u'Mo_9', u'Mo_11', u'Mo_10'],
          dtype='object')



## Linear Regression


```python
reg = linear_model.LinearRegression(normalize=True)
reg.fit(x_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)




```python
len(pred)
```




    135




```python
reg.coef_
```




    array([  6.62755296e+01,   2.04249966e+01,   1.20935659e+02,
             6.47837430e+01,   3.58486420e+01,  -3.69391903e+01,
            -3.01933766e+01,  -2.22099310e+01,  -9.81411164e+00,
            -1.91510126e+01,  -5.87872308e+00,  -7.15162930e+00,
            -1.60445105e+01,   7.74818880e+01,   1.07194064e+02,
            -5.88115195e-01,   2.49142215e+01,   2.58633358e+01,
             4.69715758e+01,   4.78092128e+00,   1.18189111e+01,
             1.19654452e+01,   3.45213517e+01,   2.97217808e+01,
             2.58060841e+01,  -8.82700496e+00,   3.86708467e+01,
             1.76495671e+00,   2.43896549e+01,  -2.68267979e-14,
             9.09873691e+01,   8.90868100e+01,  -1.18923199e+02,
            -1.26603031e+01,  -8.41581204e+00,  -9.03806384e+01,
            -8.31484193e+01,  -1.06783529e+02,  -9.10567480e+01,
            -1.11633228e+02,  -1.28826344e+02])



#### Model Validation


```python
pred = reg.predict(x_test)
pred_train = reg.predict(x_train)
```


```python
testScore = r2_score(y_pred=pred,y_true=y_test)
trainScore = r2_score(y_pred=pred_train,y_true=y_train)
```


```python
rfResults = pd.DataFrame()
rfResults['Score'] = [trainScore,testScore]
rfResults['Step'] = ['train','test']
rfResults
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>Step</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.216025</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.046039</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pointplot(y=rfResults['Score'],x=rfResults['Step'])
plt.ylim([-.1,1])
plt.title('R^2 Scores')
plt.savefig('LRScores.png',bbox_inches='tight')
```


![png](output_92_0.png)


The linear regression model performed poorly overall. 
- Slight overfitting: .216 r^2 in the train set to 0.046 r2 in the test set 

#### Let's try a different approach- a Random Forest Regressor.

## Random Forest Regression


```python
from sklearn.ensemble import RandomForestRegressor
```

For the Random Forest we will use a min samples split of 10, as to avoid overfitting.


```python
rf = RandomForestRegressor(n_estimators=500,min_samples_split=10)
rf.fit(x_train,y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=10, min_weight_fraction_leaf=0.0,
               n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False)



#### Model Validation


```python
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
predicted_train = rf.predict(x_train)
predicted_test = rf.predict(x_test)
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)

print('Test data R-2 score: {}').format(test_score)
print('Test data Spearman correlation: {}').format(spearman[0])
print('Test data Pearson correlation: {}').format(pearson[0])

train_score = r2_score(y_train, predicted_train)
spearmanTrain = spearmanr(y_train, predicted_train)
pearsonTrain = pearsonr(y_train, predicted_train)

print(' ')

print('Train data R-2 score: {}').format(train_score)
print('Train data Spearman correlation: {}').format(spearmanTrain[0])
print('Train data Pearson correlation: {}').format(pearsonTrain[0])
```

    Test data R-2 score: 0.135590561118
    Test data Spearman correlation: 0.448279100475
    Test data Pearson correlation: 0.37984956077
     
    Train data R-2 score: 0.594805178695
    Train data Spearman correlation: 0.828225597588
    Train data Pearson correlation: 0.817565023318



```python
RFperf = pd.DataFrame()
RFperf['Score'] = [train_score,test_score]
RFperf['Step'] = ['train','test']
RFperf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>Step</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.594805</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.135591</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pointplot(y=RFperf['Score'],x=RFperf['Step'],color='Red')
plt.ylim([-.1,1])
plt.title('R^2 Scores')
plt.savefig('RFScores.png',bbox_inches='tight')
```


![png](output_101_0.png)


We had solid performance in the test set, with:
    - .59 R^2 value
    - .819 Spearman Correlation

But the model fell apart when using the test set, showing clear signs of overfitting:
    - .13 R^2 value
    - .447 Spearman Correlation

#### Feature Importance


```python
predicted_test = rf.predict(x_test)

fI = pd.DataFrame()
fI['Variable'] = list(x_train.columns)
fI['Importance'] = rf.feature_importances_
fI.sort_values(by='Importance',ascending=False)[0:15]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Page total likes</td>
      <td>0.207723</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cat_1</td>
      <td>0.107980</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paid</td>
      <td>0.048507</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Mo_10</td>
      <td>0.044828</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sa</td>
      <td>0.043304</td>
    </tr>
    <tr>
      <th>26</th>
      <td>hr__13</td>
      <td>0.041345</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tu</td>
      <td>0.030655</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Mo_7</td>
      <td>0.030090</td>
    </tr>
    <tr>
      <th>17</th>
      <td>hr__4</td>
      <td>0.029611</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Th</td>
      <td>0.027883</td>
    </tr>
    <tr>
      <th>23</th>
      <td>hr__10</td>
      <td>0.026175</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Fr</td>
      <td>0.025307</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Mo_5</td>
      <td>0.021988</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cat_2</td>
      <td>0.021105</td>
    </tr>
    <tr>
      <th>22</th>
      <td>hr__9</td>
      <td>0.020185</td>
    </tr>
  </tbody>
</table>
</div>




```python
topVars= list(fI.sort_values(by='Importance',ascending=False)[0:15]['Variable'])
topVars
```




    ['Page total likes',
     'Cat_1',
     'Paid',
     'Mo_10',
     'hr__13',
     'Sa',
     'Mo_7',
     'Th',
     'hr__10',
     'hr__4',
     'Tu',
     'Fr',
     'Video',
     'Cat_2',
     'hr__1']



#### Can we improve the model by using only the top 15 features?


```python
x = df[topVars]
```


```python
x_train,x_test,y_train, y_test = train_test_split(x,
                                                  y, test_size=0.3,
                                                  random_state=50)
```


```python
rf = RandomForestRegressor(n_estimators=500,min_samples_split=15)
rf.fit(x_train,y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=15, min_weight_fraction_leaf=0.0,
               n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False)




```python
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
predicted_train = rf.predict(x_train)
predicted_test = rf.predict(x_test)
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)

#print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
print('Test data R-2 score: {}').format(test_score)
print('Test data Spearman correlation: {}').format(spearman[0])
print('Test data Pearson correlation: {}').format(pearson[0])

train_score = r2_score(y_train, predicted_train)
spearmanTrain = spearmanr(y_train, predicted_train)
pearsonTrain = pearsonr(y_train, predicted_train)

print(' ')

#print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
print('Train data R-2 score: {}').format(train_score)
print('Train data Spearman correlation: {}').format(spearmanTrain[0])
print('Train data Pearson correlation: {}').format(pearsonTrain[0])
```

    Test data R-2 score: 0.0735779002202
    Test data Spearman correlation: 0.366331840555
    Test data Pearson correlation: 0.30490941908
     
    Train data R-2 score: 0.463470449251
    Train data Spearman correlation: 0.709508194683
    Train data Pearson correlation: 0.71481692673


The model performed substantially worse when taking the top 15 features by importance from the old model. 

### Modeling Conclusion

After iterating through a random forest using the most important variables and seeing no improvement, this suggests that the data here is not rich enough to sufficiently predict likes based only on the information here.  

## Takeaways

1. Paid posts on average have higher engagement 
2. Paid Status/Photo and Category 1 and 2 posts have the highest improvement so
    - more paid posts of these types
3. Video Posts have the highest average engagement, and its not close 
	- more Video posts
4. Posting Monday in the mid-morning and Wednesday in the early morning 
    - Identify and post during times of high engagement
5. Place importance on increasing Page Likes 
    - higher probability that a post has “high” engagement
Need a richer dataset to predict accurately
    - identify potentially useful data points and collect them



## References


(Moro et al., 2016) Moro, S., Rita, P., & Vala, B. (2016). Predicting social media performance metrics and evaluation of the impact on brand building: A data mining approach. Journal of Business Research, 69(9), 3341-3351. 

