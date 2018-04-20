# Authors: Sandesh Swamy, Alan Ritter, and Marie-Catherine de Marneffe
# Copyright, 2017 
# Demo for paper in EMNLP 2017.
from sklearn.linear_model import LogisticRegression
import numpy as np
hash = {
	'DY': 3,
	'PY': 3,
	'UC': 2,
	'PN': 1,
	'DN': 1,
}
yvec = []
ydata = open('../data/ydata.txt', 'r')
lines = ydata.readlines()
for item in lines:
	yvec.append(hash[item.strip()])
yvec = np.array(yvec)
print yvec
print yvec.shape
xmat = np.load('../data/xdata.npy')
print xmat.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xmat, yvec, test_size=0.3, random_state=42)
lrmulti = LogisticRegression(solver='lbfgs', multi_class='multinomial')
lrmulti.fit(X_train, y_train)
y_pred = lrmulti.predict(X_test)
from sklearn.metrics import *
print classification_report(y_test, y_pred)
