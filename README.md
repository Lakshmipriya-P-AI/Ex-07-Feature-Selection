# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file.

## Explanation
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

## ALGORITHM
### STEP 1
Read the given Data

### STEP 2
Clean the Data Set using Data Cleaning Process

3## STEP 3
Apply Feature selection techniques to all the features of the data set

### STEP 4
Save the data to the file

## CODE
```
Developed by: Lakshmi Priya P
Register number: 212221230053

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()

df.columns = boston['feature_names']
df.head()

df['PRICE']= boston['target']
df.head()

df.info()

plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()

#FILTER METHODS

X=df.drop("PRICE",1)
y=df["PRICE"]

from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape

#1.Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)

#2.Information gain/Mutual Information
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

#3.SelectKBest Model
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2) 
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))

#4.Correlation Coefficient
cor=df.corr()
sns.heatmap(cor,annot=True)

#5.Mean Absolute Difference
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='purple')

#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)

#6.Chi Square Test
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

#7.SelectPercentile method
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape

#WRAPPER METHOD

#1.Forward feature selection

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X, y)
sfs.k_feature_names_

#2.Backward feature elimination

sbs = SFS(LinearRegression(),
         k_features=10,
         forward=False,
         floating=False,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#3.Bi-directional elimination

sffs = SFS(LinearRegression(),
         k_features=(3,7),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_

#4.Recursive Feature Selection

from sklearn.feature_selection import RFE
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=7)
rfe.fit(X, y)
print(X.shape, y.shape)
rfe.transform(X)
rfe.get_params(deep=True)
rfe.support_
rfe.ranking_

#EMBEDDED METHOD

#1.Random Forest Importance

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X,y_transformed)
importances=model.feature_importances_

final_df=pd.DataFrame({"Features":pd.DataFrame(X).columns,"Importances":importances})
final_df.set_index("Importances")
final_df=final_df.sort_values("Importances")
final_df.plot.bar(color="purple")
```
## OUPUT
<img width="502" alt="170059652-ecac492d-58e1-469d-8ed8-80fc6b996a09" src="https://user-images.githubusercontent.com/93427923/170067199-7b40efaf-584f-4e65-9e13-d709ef9758e4.png">

### Analyzing the boston dataset:
<img width="639" alt="170059739-2ca715f7-f9e7-4f73-82a5-a659df046120" src="https://user-images.githubusercontent.com/93427923/170067266-a19d67ed-41cd-46d5-b47f-35ea04fb2a2d.png">
<img width="593" alt="170059846-e16f0c4d-9c7e-4c2b-9bdf-8d1263aa4e0f" src="https://user-images.githubusercontent.com/93427923/170067355-f6714ede-d9fd-482b-ba73-1ce6626b8bd3.png">
<img width="572" alt="170060583-ce248897-0175-446b-9631-277287c30310" src="https://user-images.githubusercontent.com/93427923/170067397-00397dfa-0f5d-4ea3-a9fa-40b8adae86c9.png">

### Analyzing dataset using Distplot:
<img width="658" alt="170062051-3b4b6994-44f1-45f0-8aad-3301ab65ac45" src="https://user-images.githubusercontent.com/93427923/170067462-7e910061-9e75-4615-a69a-9bfe7c8b1205.png">

## Filter Methods:
### Variance Threshold:
![170062195-342db6b6-5ecf-4e19-819e-d97f0b1a7612](https://user-images.githubusercontent.com/93427923/170067587-0d4e96e6-f6c7-4db2-9a56-58dd7063093d.png)

### Information Gain:
![170062251-9208af20-79b9-42cf-8c3e-62441b1eccec](https://user-images.githubusercontent.com/93427923/170067700-5f6ea7e3-4410-4d27-a96b-eaa0cd053fdc.png)

### SelectKBest Model:
![170062297-51f0d29a-2612-43fb-8001-d9ce8c9748ec](https://user-images.githubusercontent.com/93427923/170067807-eb1e447c-b64a-41ef-b616-365f222d11e1.png)

### Correlation Coefficient:
![170062340-9db2c60b-59c0-4398-a1da-aedc03f70348](https://user-images.githubusercontent.com/93427923/170067879-63b15be1-d71b-472a-83f5-063ec99200a0.png)

### Mean Absolute difference:
![170062388-27505cf5-8eee-4b08-8950-a6bf6b06cb73](https://user-images.githubusercontent.com/93427923/170067959-f6e3aa39-63cc-4650-8774-4c49a6f6175f.png)

### Chi Square Test:
![170062449-44c326c2-9e23-4a2a-aa3b-440ac3b3132a](https://user-images.githubusercontent.com/93427923/170068046-db80a17b-31c6-4451-8675-87a61b3b5c98.png)
![170062483-af539458-30b8-4bf8-9a7c-0c049889ea6d](https://user-images.githubusercontent.com/93427923/170068078-7ee4b1b8-5934-48b1-b0a1-aa9f42e5a3fd.png)

### SelectPercentile Method:
![170062524-e6ee2f3e-8f49-452d-93b2-f55cc6eadbc2](https://user-images.githubusercontent.com/93427923/170068149-ffc835a0-1280-4aab-a21c-fed9e35edf93.png)

## Wrapper Methods:
### Forward Feature Selection:
![170062678-726e9650-9403-455a-9f8c-d486f6ae4df2](https://user-images.githubusercontent.com/93427923/170068275-33326db0-b028-4db8-b08f-15953c130372.png)

### Backward Feature Selection:
![170062722-6c961a8e-c74a-45e0-9347-acc686c7131f](https://user-images.githubusercontent.com/93427923/170068574-03b200fc-ca19-4db3-8d3a-4810f728836a.png)

### Bi-Directional Elimination:
![170062760-45a2e6af-086b-4f97-901e-6d09094f828c](https://user-images.githubusercontent.com/93427923/170068638-59c1a737-efa1-4315-9b98-cd42c18285be.png)

### Recursive Feature Selection:
![170062794-840668a5-a434-4830-a23c-d87cd76afcc4](https://user-images.githubusercontent.com/93427923/170068742-3c7fb989-1fc0-4b34-a3c0-e673b0710a6b.png)

## Embedded Methods:
### Random Forest Importance:
![170062874-d3c99008-345f-4e94-9370-1bba686e7440](https://user-images.githubusercontent.com/93427923/170068787-2a957043-7074-434d-96be-bac00f1f01f3.png)

## RESULT:
Hence various feature selection techniques are applied to the given data set successfully and saved the data into a file.
