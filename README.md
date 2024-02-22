# Final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

d = pd.read_csv('googleplaystore.csv')
# there is an unusual row that should be removed
d.drop(index=d[d['Reviews']=='3.0M'].index,inplace=True)
# there are rows with values that are not informative
np.sum(d=='Varies with device')
# these not informative lines are dropped
d.drop(index=d[d['Size']=='Varies with device'].index,inplace=True)
d.drop(index=d[d['Current Ver']=='Varies with device'].index,inplace=True)
d.drop(index=d[d['Android Ver']=='Varies with device'].index,inplace=True)
# some numerical values are saved as strings that should be converted
d['Reviews'] = d['Reviews'].astype('int')
def func1(x):
    if 'M' in x:
        return 1e6*float(x[:-1])
    elif 'k' in x:
        return 1e3*float(x[:-1])
    else:
        return float(x)

d['Size'] = d['Size'].apply(func1)
d['Installs'] = d['Installs'].apply(lambda x:int(x[:-1].replace(',','')))
d['Price'] = d['Price'].apply(lambda x:int(x) if x=='0' else float(x[1:]))
# some feature are not relevant, so they are removed
d.drop(columns=['App','Current Ver','Android Ver','Last Updated'],inplace=True)

imp = SimpleImputer(strategy='most_frequent')
d['Rating'] = imp.fit_transform(d['Rating'].to_numpy().reshape(-1,1))
d['high rating'] = (d['Rating']>3.9).astype('int')

fig,ax = plt.subplots(2,2,figsize=[12,10])
sns.barplot(x='high rating',y='Reviews',data=d, ax=ax[0,0])
sns.barplot(x='high rating',y='Installs',data=d, ax=ax[0,1])
sns.barplot(x='high rating',y='Size',data=d, ax=ax[1,0])
sns.barplot(x='high rating',y='Price',data=d, ax=ax[1,1])

fig,ax = plt.subplots(3,1,figsize=[15,15])
sns.countplot(x='Genres',data=d,hue='high rating',ax=ax[0])
sns.countplot(x='Category',data=d,hue='high rating', ax=ax[1])
sns.countplot(x='Type',data=d,hue='high rating', ax=ax[2])

fig,ax = plt.subplots(2,2,figsize=[15,12])
sns.barplot(x='Category',y='Rating',data=d, ax=ax[0,0])
sns.barplot(x='Genres',y='Rating',data=d, ax=ax[0,1])
sns.barplot(x='Type',y='Rating',data=d, ax=ax[1,0])
sns.barplot(x='Content Rating',y='Rating',data=d, ax=ax[1,1])

fig,ax = plt.subplots(2,2,figsize=[15,12])
sns.scatterplot(x='Reviews',y='Rating',data=d, ax=ax[0,0])
sns.scatterplot(x='Installs',y='Rating',data=d, ax=ax[0,1])
sns.scatterplot(x='Size',y='Rating',data=d, ax=ax[1,0])
sns.scatterplot(x='Price',y='Rating',data=d, ax=ax[1,1])

sns.pairplot(data=d,hue='high rating')

def ordinalEncoder(df):
    encoder = OrdinalEncoder()
    d = df.select_dtypes(include=['bool','object'])
    d.loc[:,:] = encoder.fit_transform(d)
    return pd.concat([df.select_dtypes(exclude=['bool','object']),d],axis=1)
y = d['high rating']
X = ordinalEncoder(d.drop(columns=['Rating','high rating']))

def chiFeaSelect(X,y,s):
    stats,pval = chi2(X,y)
    return X.loc[:,pval<s]
X = chiFeaSelect(X,y,0.005)

# simple train test method
# scaler initialization
scaler = StandardScaler()

# train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# scaling only train data (to avoid data leakage)
X_train = scaler.fit_transform(X_train)

# models definition
model1 = RandomForestClassifier(n_estimators=200)
model2 = LogisticRegression()
model3 = SVC(kernel='linear')
model4 = KNeighborsClassifier(n_neighbors=5)
model5 = DecisionTreeClassifier()

# models training
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
model4.fit(X_train,y_train)
model5.fit(X_train,y_train)

# models testing
y_pred1 = model1.predict(scaler.transform(X_test))
y_pred2 = model2.predict(scaler.transform(X_test))
y_pred3 = model3.predict(scaler.transform(X_test))
y_pred4 = model4.predict(scaler.transform(X_test))
y_pred5 = model5.predict(scaler.transform(X_test))

# models performance report
print(f'classification report for random forest model is:\n {classification_report(y_test,y_pred1)}')
print(f'confusion matrix is:\n {confusion_matrix(y_test,y_pred1)}')
print(f'classification report for logistic regression model is:\n {classification_report(y_test,y_pred2)}')
print(f'confusion matrix is:\n {confusion_matrix(y_test,y_pred2)}')
print(f'classification report for SVM model is:\n {classification_report(y_test,y_pred3)}')
print(f'confusion matrix is:\n {confusion_matrix(y_test,y_pred3)}')
print(f'classification report for KNN is:\n {classification_report(y_test,y_pred4)}')
print(f'confusion matrix is:\n {confusion_matrix(y_test,y_pred4)}')
print(f'classification report for decision tree model is:\n {classification_report(y_test,y_pred5)}')
print(f'confusion matrix is:\n {confusion_matrix(y_test,y_pred5)}')

# model testing with cross validation
# scaling whole features
scaler = StandardScaler()
X1 = scaler.fit_transform(X)

# models definition
model1 = RandomForestClassifier(n_estimators=200)
model2 = LogisticRegression()
model3 = SVC(kernel='linear')
model4 = KNeighborsClassifier(n_neighbors=5)
model5 = DecisionTreeClassifier()

# cross validating with 5-fold structure
y_pred1 = cross_val_predict(model1,X1,y,cv=5)
y_pred2 = cross_val_predict(model2,X1,y,cv=5)
y_pred3 = cross_val_predict(model3,X1,y,cv=5)
y_pred4 = cross_val_predict(model4,X1,y,cv=5)
y_pred5 = cross_val_predict(model5,X1,y,cv=5)

# models performance report
print(f'classification report for random forest model is:\n {classification_report(y,y_pred1)}')
print(f'confusion matrix is:\n {confusion_matrix(y,y_pred1)}')
print(f'classification report for logistic regression model is:\n {classification_report(y,y_pred2)}')
print(f'confusion matrix is:\n {confusion_matrix(y,y_pred2)}')
print(f'classification report for SVM model is:\n {classification_report(y,y_pred3)}')
print(f'confusion matrix is:\n {confusion_matrix(y,y_pred3)}')
print(f'classification report for KNN is:\n {classification_report(y,y_pred4)}')
print(f'confusion matrix is:\n {confusion_matrix(y,y_pred4)}')
print(f'classification report for decision tree model is:\n {classification_report(y,y_pred5)}')
print(f'confusion matrix is:\n {confusion_matrix(y,y_pred5)}')

# Regression without scaling
# train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# model definition
model1 = LinearRegression()
model2 = SGDRegressor()
model3 = GradientBoostingRegressor()

# model training
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)

# model testing
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

print(f'linear regression model score is:\n{model1.score(X_test,y_test)}')
print(f'MSE is:\n{mean_squared_error(y_test,y_pred1,squared=False)}\n')

print(f'stochastic gradient descent model score is:\n{model2.score(X_test,y_test)}')
print(f'MSE is:\n{mean_squared_error(y_test,y_pred2,squared=False)}\n')

print(f'gradient boosting regression model score is:\n{model3.score(X_test,y_test)}')
print(f'MSE is:\n{mean_squared_error(y_test,y_pred3,squared=False)}\n')


# Regression with scaling
# Regression without scaling
# train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# model definition
model1 = LinearRegression()
model2 = SGDRegressor()
model3 = GradientBoostingRegressor()

# scaling only train data (to avoid data leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# model training
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)

# model testing
y_pred1 = model1.predict(scaler.transform(X_test))
y_pred2 = model2.predict(scaler.transform(X_test))
y_pred3 = model3.predict(scaler.transform(X_test))

print(f'linear regression model score is:\n{model1.score(X_test,y_test)}')
print(f'MSE is:\n{mean_squared_error(y_test,y_pred1,squared=False)}\n')

print(f'stochastic gradient descent model score is:\n{model2.score(X_test,y_test)}')
print(f'MSE is:\n{mean_squared_error(y_test,y_pred2,squared=False)}\n')

print(f'gradient boosting regression model score is:\n{model3.score(X_test,y_test)}')
print(f'MSE is:\n{mean_squared_error(y_test,y_pred3,squared=False)}\n')
