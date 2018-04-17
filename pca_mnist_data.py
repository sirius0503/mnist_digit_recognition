import pandas as pd

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='/home/aspiring1/Private/python_data_science') #, 

# why is data_home required

from sklearn.model_selection import train_test_split

# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# Applying kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 80, kernel = 'rbf')
train_img = kpca.fit_transform(train_img)
train_img = kpca.transform(train_img)

pca.fit(train_img)

pca.n_components_

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

from sklearn.linear_model import LogisticRegression

#all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')

logisticRegr.fit(train_img, train_lbl)

# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))

#Predict for One Observation (image)
logisticRegr.predict(test_img[0:10])

logisticRegr.score(test_img, test_lbl)

from sklearn import svm



clf = svm.SVC()
clf.fit(train_img, train_lbl)
clf.score(test_img, test_lbl)

#pca1 = PCA(n_components = 305)
#test_img1 = mnist.data[42000:]
#pca1.fit(test_img1)
#test_img1 = pca1.transform(test_img1)
results = clf.predict(test_img)

df = pd.DataFrame(results)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results1.csv', header=True)

# Fitting Xgboost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(train_img, train_lbl)

# Predicting the Test set results
y_pred = classifier.predict(test_img)

# Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = train_img, y = train_lbl, cv = 10)
accuracies.mean()
accuracies.std()

explained_variance[0:80].sum()

# Applying xgboost algorithm
# Fitting Xgboost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(train_img, train_lbl)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
classifier.score(test_img, test_lbl)
