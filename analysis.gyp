import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

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
#print(depression.columns)

control_frontal = control[['AB.A.delta.a.FP1', 'AB.A.delta.b.FP2']]
depression_frontal = depression[['AB.A.delta.a.FP1', 'AB.A.delta.b.FP2']] # this is just FP1 and FP2

# Kernal Density Estimation
# The KDE for FP1/FP2 show a tendency for MDD pts. to have frequency --> excessive introspection?
kde_d = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(depression_frontal)
kde_c = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(control_frontal)
kde_depression = kde_d.score_samples(depression_frontal)
kde_control = kde_c.score_samples(control_frontal)
kde_fig = plt.figure()
ax0 = kde_fig.add_subplot(111)
ax0.plot(kde_control, c = 'b', label = 'depression')
ax0.plot(kde_depression, c = 'r', label = 'control')
plt.xlabel("Wave Frequency")
plt.ylabel("KDE Score")
plt.xlim(0, 60)
plt.title("KDE: Delta Wave, FP1 and FP2")
plt.legend(loc = 'upper left')
#plt.show()

depression_beta = depression[['AB.D.beta.a.FP1', 'AB.D.beta.b.FP2']] # this is just FP1 and FP2
control_beta = control[['AB.D.beta.a.FP1', 'AB.D.beta.b.FP2']]

depression_theta = depression[['AB.B.theta.a.FP1', 'AB.B.theta.b.FP2']]
control_theta = control[['AB.B.theta.a.FP1', 'AB.B.theta.b.FP2']]

depression_alpha = depression[['AB.C.alpha.a.FP1', 'AB.C.alpha.b.FP2']]
control_alpha = control[['AB.C.alpha.a.FP1', 'AB.C.alpha.b.FP2']]

# Simple plotting of FP1 and FP2
# plot also tells same story as KDE
scatter_fig1 = plt.figure()
ax1 = scatter_fig1.add_subplot(111)
ax1.scatter(depression_frontal.iloc[:,0], depression_frontal.iloc[:,1], c = 'b', label = 'depression')
ax1.scatter(control_frontal.iloc[:,0], control_frontal.iloc[:,1], c = 'r', label = 'control')
plt.xlabel("Delta Wave FP1")
plt.ylabel("Delta Wave FP2")
plt.legend(loc = 'upper left')
#plt.show()

scatter_fig2 = plt.figure()
ax1 = scatter_fig2.add_subplot(111)
ax1.scatter(depression_beta.iloc[:,0], depression_beta.iloc[:,1], c = 'b', label = 'depression')
ax1.scatter(control_beta.iloc[:,0], control_beta.iloc[:,1], c = 'r', label = 'control')
plt.xlabel("Beta Wave FP1")
plt.ylabel("Beta Wave FP2")
plt.legend(loc = 'upper left')


scatter_fig3 = plt.figure()
ax1 = scatter_fig3.add_subplot(111)
ax1.scatter(depression_theta.iloc[:,0], depression_theta.iloc[:,1], c = 'b', label = 'depression')
ax1.scatter(control_theta.iloc[:,0], control_theta.iloc[:,1], c = 'r', label = 'control')
plt.xlabel("Theta Wave FP1")
plt.ylabel("Theta Wave FP2")
plt.legend(loc = 'upper left')

scatter_fig4 = plt.figure()
ax1 = scatter_fig4.add_subplot(111)
ax1.scatter(depression_alpha.iloc[:,0], depression_alpha.iloc[:,1], c = 'b', label = 'depression')
ax1.scatter(control_alpha.iloc[:,0], control_alpha.iloc[:,1], c = 'r', label = 'control')
plt.xlabel("Alpha Wave FP1")
plt.ylabel("Alpha Wave FP2")
plt.legend(loc = 'upper left')
plt.show()





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





