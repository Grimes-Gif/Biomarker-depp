import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree





data = pd.read_csv('/Users/saifali/Desktop/School/CS 4641/Project/eeg_data.csv')

# print("---------\nKEY: \n1: MDD\n2: Schizophrenia\n3: Contol\n4: Substance Abuse\n5: Behavioral Addiction\n---------")
# 945 rows by 1149 columns
# WAVE TYPE.ELECTRODE TYPE


# replace the string values with int values: 1 -> MDD, 
# 2 -> schizophrenia, 3 -> Control, 4 -> Substance abuse, 
# 5 -> Behavioral addiction
# oldValues: depressive disorder, healthy control, etc
# newValues: 1, 2, 3, etc
# label: just disorders
def replace_values(data, label, oldValues, newValues):
    if len(oldValues) != len(newValues):
        raise Exception("Inconsistent replacement")

    for indexOf, value in enumerate(oldValues):
        data[label] = data[label].replace(value, newValues[indexOf])

    return data

#Get a specific disorder and compare it to the control across all electrodes
def disorder_vs_control_electrode(data, label, disorders, control):
    newData = []
    for value in disorders:
        #print(value)
        a = data.loc[data[label].isin([value, control])]
        newData.append(a)
        #print(newData)
    return newData

#Sum up same wave frequencies column wise across all sensors
def get_summedFrequency(data, wave_types):
    for wave in wave_types:
        pass

def run_pca(dataset, n_components):
    pca = PCA(n_components)
    pca.fit_transform(dataset) # this should work because python passes by reference 
    return pca
    

def graph(thing):
    plt.plot(thing)
    plt.show()

def max_(num):
    pd.set_option('display.max_columns', num)
    pd.set_option('display.max_rows', num)

# init useful lists for cleaning
labels = ['Depressive disorder', 'Schizophrenia', 'Healthy control', 'Alcohol use disorder', 'Behavioral addiction disorder']
label_map = [1, 2, 3, 4, 5]
disorders = ['Depressive disorder', 'Schizophrenia', 'Alcohol use disorder', 'Behavioral addiction disorder']
frivolous_data = ['sex', 'age', 'eeg.date', 'education', 'IQ', 'main.disorder', 'no.'] # remove unwanted features

# clean data
cleaned_data = data.drop(frivolous_data, axis = 1) # drop columns we don't want
cleaned_data = replace_values(cleaned_data, 'specific.disorder', labels, label_map) # replace disorders w/ numbers

# take all coherence data out
# try to run pca on this
freq_data = cleaned_data[cleaned_data.columns.drop(list(cleaned_data.filter(regex='COH')))]
freq_data = freq_data[freq_data.columns.drop(list(freq_data.filter(regex='Unnamed')))]
#print(freq_data)




#cleaned_data = cleaned_data.drop("specific.disorder", axis=1)
#cleaned_data_pca = run_pca(cleaned_data, 25)
#print("Recovered (total): ", np.sum(cleaned_data_pca.explained_variance_ratio_))


# create separate datasets for each disorder
depression = cleaned_data[cleaned_data['specific.disorder'] == 1]
schizophrenia = cleaned_data[cleaned_data['specific.disorder'] == 2]
control = cleaned_data[cleaned_data['specific.disorder'] == 3]
substance_abuse = cleaned_data[cleaned_data['specific.disorder'] == 4]
behavioral_addiction = cleaned_data[cleaned_data['specific.disorder'] == 5]

# clean them up so they are matching
control = control.reset_index()
depression = depression.reset_index()
depression = depression.truncate(after = 94, axis = 0) # truncate depression so u can compare w/ control

# removed NaN
depression = depression.fillna(0) 
control = control.fillna(0)

# recovery is ~80% for both - acceptable
pca_depression = run_pca(depression, 12)
pca_control = run_pca(control, 12)
#print("Total Recovered Depression: ", np.sum(pca_depression.explained_variance_ratio_))
#print("Total Recovered Control: ", np.sum(pca_control.explained_variance_ratio_))
max_(None)
#print(depression)

control_frontal = control[['AB.A.delta.a.FP1', 'AB.A.delta.b.FP2']]
depression_frontal = depression[['AB.A.delta.a.FP1', 'AB.A.delta.b.FP2']] # this is just FP1 and FP2





# generate schizophrenia vs control dataset: beta waves, all frontal electrodes
control_fullfrontal = control[['specific.disorder', 'AB.D.beta.a.FP1', 'AB.D.beta.b.FP2', 'AB.D.beta.c.F7', 'AB.D.beta.d.F3',
                                'AB.D.beta.e.Fz', 'AB.D.beta.f.F4', 'AB.D.beta.g.F8']]

schiz_fullfrontal = schizophrenia[['specific.disorder', 'AB.D.beta.a.FP1', 'AB.D.beta.b.FP2', 'AB.D.beta.c.F7', 'AB.D.beta.d.F3',
                                'AB.D.beta.e.Fz', 'AB.D.beta.f.F4', 'AB.D.beta.g.F8']]


# now use supervised learning to train a model based on schizohprenia training data, and then apply to test data
# and then do comparisons between schiz and control
# remember these are for BETA waves
schiz_fullfrontal = schiz_fullfrontal.reset_index()
control_fullfrontal = control_fullfrontal.reset_index()

schiz_fullfrontal.fillna(0)
control_fullfrontal.fillna(0)

schiz_fullfrontal = schiz_fullfrontal.drop('index', axis = 1)
control_fullfrontal = control_fullfrontal.drop('index', axis = 1)

full_frontal = pd.concat([control_fullfrontal, schiz_fullfrontal], axis=0)
full_frontal = full_frontal.reset_index()
full_frontal = full_frontal.drop('index', axis=1)

full_frontal = shuffle(full_frontal)
full_frontal.reset_index(inplace=True)
full_frontal = full_frontal.drop('index', axis=1)




labels = full_frontal['specific.disorder']
labels = labels.tolist()

# split set
train, test, train_labels, test_labels = train_test_split(full_frontal, labels, test_size = 0.33)


# make labels
#train_labels = labels[:142]
#test_labels = labels[142:]


#print(train_labels)
#print(train)

#print(test_labels)
#print(test)

# running gaussian:
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
pred = model.predict(test)
#print(accuracy_score(test_labels, pred))

# sgd - average accuracy of 71%
sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=15)
sgd.fit(train, train_labels)
pred = sgd.predict(test)
print("SGD: ", accuracy_score(test_labels, pred))

# Multi-Layer Perceptron - 57% avg
# didn't work because neural networks require more data to train
neuralnet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
neuralnet.fit(train, train_labels)
pred = neuralnet.predict(test)
print("MLP: ", accuracy_score(test_labels, pred))
"""
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = pred.coefs_[0].min(), pred.coefs_[0].max()
for coef, ax in zip(pred.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
"""
# SVM linear kernal - 55% 
# SVM w/ polynomial - 70% 
C = 1
sv = svm.SVC(kernel='poly', degree=3, C=C).fit(train, train_labels)
pred = sv.predict(test)
print("SVM: ", accuracy_score(test_labels, pred))

X = train
X = X.drop('specific.disorder', axis=1)

# Average of 55% 
tc = tree.DecisionTreeClassifier()
tc = tc.fit(train, train_labels)
tc.predict(test)
plt.figure()
tree.plot_tree(tc)
plt.title("Decision tree")
plt.show()
print("Tree: ", accuracy_score(test_labels, pred))

# decision trees overfit data with too many features
full = pd.concat([control, depression], axis=0)

full = full.fillna(0)
full = full.reset_index()
full = full.drop('index', axis=1)
full_labels = full['specific.disorder']
full_labels = full_labels.tolist()
train2, test2, train_labels2, test_labels2 = train_test_split(full, full_labels, test_size = .33)

sv.fit(train2, train_labels2)
pred = sv.predict(test2)
print("Full data (Neural Network): ", accuracy_score(test_labels2, pred))



"""
# Charting SVM
X = train.iloc[1:,:3]
y = train_labels

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot in
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('FP1')
    plt.ylabel('FP2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

#print(schiz_fullfrontal)
#print(control_fullfrontal)

# split data into training and testing









# testing
#print("Depression: \n", depression)
#print("Schizophrenia: \n", schizophrenia)
#print("Control: \n", control)
#print("Substance Abuse: \n", substance_abuse)
#print("Behavioral Addiction: \n", behavioral_addiction)

"""

"""
# not great results...
kmeans = KMeans(n_clusters = 3)
label = kmeans.fit_predict(depression)

filtered_label0 = depression[label == 0]
filtered_label1 = depression[label == 1]
filtered_label2 = depression[label == 2]

print(filtered_label0)
print(filtered_label1)

plt.scatter(filtered_label0.iloc[:,0] , filtered_label0.iloc[:,1])
plt.scatter(filtered_label1.iloc[:,0] , filtered_label1.iloc[:,1])
plt.scatter(filtered_label2.iloc[:,0] , filtered_label2.iloc[:,1])
#plt.show()
"""



#cleaned_unlabeled = cleaned_data.drop('specific.disorder', axis=1)
#print(cleaned_unlabeled.head())
"""
"""