import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import StandardScaler

def solve(XX, eig, feature, dimension):     # Projection Onto the New Feature Space

	if (dimension == 1):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1)))

	elif (dimension == 2):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1), eig[1][1].reshape(feature,1)))

	elif(dimension == 3):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1), eig[1][1].reshape(feature,1), eig[2][1].reshape(feature,1)))

	elif(dimension == 4):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1), eig[1][1].reshape(feature,1), eig[2][1].reshape(feature,1), eig[3][1].reshape(feature,1)))

	elif(dimension == 5):
		matrix_w = np.hstack((eig[0][1].reshape(feature,1), eig[1][1].reshape(feature,1), eig[2][1].reshape(feature,1), eig[3][1].reshape(feature,1), eig[4][1].reshape(feature,1)))

	print('Matrix W:\n', matrix_w)

	Y = XX.dot(matrix_w)

	return Y;


# *************************************************** Dataset  *******************************************************
train = pd.read_csv('train.txt')
test = pd.read_csv('test.txt')
train.drop(['id', 'date'], 1, inplace = True)
test.drop(['id', 'date'], 1, inplace = True)

print(train)
print(test)

X_train = np.array(train.drop(['Occupancy'], 1))
y_train = np.array(train['Occupancy'])

print(X_train)
print(y_train)

X_test = np.array(test.drop(['Occupancy'], 1))
y_test = np.array(test['Occupancy'])

print(X_test)
print(y_test)

# Normalization of data as different variables in data set may be having different units of measurement 
X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)
#X_train_std = X_train
#X_test_std = X_test

###############################################################################
###############################################################################
total_feature = 5
selected_dimension = 3
###############################################################################
###############################################################################


# *******************************************  Reduce dimension from training data ******************************************************************* #
mean_vec = np.mean(X_train_std, axis=0)
cov_mat = (X_train_std - mean_vec).T.dot((X_train_std - mean_vec)) / (X_train_std.shape[0] - 1)
print('Covariance matrix \n%s' %cov_mat)


cov_mat = np.cov(X_train_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort()
eig_pairs.reverse()

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

X_train_new = solve(X_train_std, eig_pairs, total_feature, selected_dimension)
print(X_train_new)


# *******************************************  Reduce dimension from test data *********************************************************************** #
mean_vec = np.mean(X_test_std, axis=0)
cov_mat = (X_test_std - mean_vec).T.dot((X_test_std - mean_vec)) / (X_test_std.shape[0] - 1)
print('Covariance matrix \n%s' %cov_mat)


cov_mat = np.cov(X_test_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort()
eig_pairs.reverse()

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

X_test_new = solve(X_test_std, eig_pairs, total_feature, selected_dimension)
print(X_test_new)


# ************************************************** Accuracy Check using SVM classifier *********************************************************** #

model = svm.SVC()
model.fit(X_train_new, y_train)

accuracy = model.score(X_test_new, y_test)
print(accuracy)


# ************************************************************ END ********************************************************************************* #
