# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# %%
import os #access operating system
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


# %%
filepath = '/home/roshankondapalli/Documents/Textbooks1/Sci_Computing/Databases/Practice_Datasets/hipparcos-voidmain.csv'
data = pd.read_csv(filepath)
data.head(10)

# %%
data.info()

# %%
for i in data:
    print(i, data[i].isnull().sum())

# %%
data['SpType'].isnull().sum()

# %%
data['SpType'].nunique()

# %%
data1 = data.copy()

# %%
#data1 = data1['SpType'].replace(['Iab|Ia|Iab|','II','III','IV''VII','VI|sd','VII','V'], 
                             #   ['Super Giant', 'Bright Giant', 'Normal Giant', 'Sub Giant', 'Sub Dwarf', 
                              #   'White Dwarf', 'Main Sequence'])

# %%
#data1 = data1[data1["SpType"].str.contains("/|-") == False]
#data1 = data1[data1['SpType'].str.contains('Super Giant|Bright Giant|Normal Giant|Sub Giant|Sub Dwarf|White Dwarf|Main Sequence') == True]

# %%
#Dropping ambiguous classifiers
data1 = data1[data1["SpType"].str.contains("/|-")==False]

#Extracting classes of interest and applying new labels
data1.loc[data1["SpType"].str.contains("VII") == True, "SpType"] = "White Dwarf"
data1.loc[data1["SpType"].str.contains("VI|sd") == True, "SpType"] = "Sub Dwarf"
data1.loc[data1["SpType"].str.contains("IV") == True, "SpType"] = "Sub Giant"
data1.loc[data1["SpType"].str.contains("V") == True, "SpType"] = "Main Sequence Star"
data1.loc[data1["SpType"].str.contains("III") == True, "SpType"] = "Normal Giant"
data1.loc[data1["SpType"].str.contains("II") == True, "SpType"] = "Bright Giant"
data1.loc[data1["SpType"].str.contains("Ia|Ib|Iab") == True, "SpType"] = "Super Giant"

#Filtering unnecessary classes
data1 = data1[data1["SpType"].str.contains("Super Giant|Normal Giant|Bright Giant|Main Sequence|Sub Giant|Sub Dwarf|White Dwarf") == True]


# %%
data1.head()

# %%
len(data1)

# %%
spectral_data = data1.loc[:,['SpType','V-I','B-V','Plx','Vmag']]
spectral_data = spectral_data.dropna()

spectral_data["Plx"] = spectral_data["Plx"].apply(abs)
spectral_data = spectral_data[spectral_data["Plx"] > 0]

spectral_data.head()


# %%
def Absolute_mag(Plx, Vmag):
    parsecs = 1000/Plx
    Amag = 0
    Amag = Vmag - 5*np.log10(parsecs/10)

    
    return Amag

# %%
spectral_data["Amag"] = spectral_data.apply(lambda x: Absolute_mag(x['Plx'], x['Vmag']), axis = 1)

spectral_data.head()

# %%
sns.pairplot(spectral_data, plot_kws=dict(alpha=.1, edgecolor='none'))
plt.show()

# %%
spectral_data = spectral_data.copy()

# %%
#spectral_data = spectral_data.drop(['Plx','Vmag'], axis = 1)

# %%
spectral_data.head()

# %%
plot_cols = ['SpType', 'V-I', 'B-V', 'Amag']

sns.pairplot(spectral_data[plot_cols], diag_kind='kde') #kde = Kernel Density Estimate

plt.show()

# %%
Pair_plot1 = sns.pairplot(spectral_data[plot_cols], hue = 'SpType')

plt.show()

# %%
Pair_plot1.savefig("Pairplot1.png")
plt.show()

# %%
JointPlot1 = sns.jointplot(data=spectral_data, x='B-V', y='Amag', hue='SpType', palette="Set2")
plt.show()

# %%
JointPlot1.savefig("Jointplot1.png")
plt.show()

# %%
spectral_data.info()

# %%
spectral_data.describe(include = 'float64')

# %%
for i in spectral_data:
    print(i, spectral_data[i].isnull().sum())

# %%
spect_data = spectral_data.copy()

# %%
spec_data = spect_data.drop(['Plx','Vmag'], axis = 1)

# %%
spec_data.head()

# %%
B_Vindex = np.array(spec_data['B-V'], dtype=float).tolist()
A_mag = np.array(spec_data['Amag'], dtype=float).tolist()

fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(B_Vindex, A_mag, cmap = "plasma", c = B_Vindex, marker = '.', s = 10)

plt.title("Hertzsprung-Russel Diagramme")
plt.xlabel("B-V Index")
plt.ylabel("Absolute Magnitude")

plt.xlim(-0.5, 3.5)
plt.ylim(17,-17)

plt.savefig("Hertzprung-Russell Diagramme1.png")
plt.show()

# %%
spec_data = spec_data.rename(columns={'V-I': 'V-I Index','B-V':'B-V Index',
                                      'Amag':'Absolute Magnitude', 'SpType':'Spectral Type' })
spec_data.head()

# %%
B_Vindex = np.array(spec_data['B-V Index'], dtype=float).tolist()
A_mag = np.array(spec_data['Absolute Magnitude'], dtype=float).tolist()

sns.set(rc={"figure.figsize":(13,13), 
            "axes.labelsize":12})

#fig, ax = plt.subplots(figsize = (10,10))
#x.scatter(B_Vindex, A_mag, cmap = "plasma", c = B_Vindex, marker = '.', s = 10)

ax = sns.scatterplot(data=spec_data, x=B_Vindex, y=A_mag, 
                     hue='Spectral Type', palette="Set2")

ax.set(xlabel='B-V Index',
       ylabel='Absolute Magnitude',
       title='Hertzsprung-Russell Diagramme')
ax.set_xlim(-0.5,3.5)
ax.set_ylim(17,-17) 

plt.savefig('Hertzsprung_Russell_Diagramme2.png')
plt.show()

# %%
fig, axes = plt.subplots(2, 2, figsize=(20,12))

sns.scatterplot(data = spectral_data, x = 'Vmag', y= 'B-V', ax = axes[0][0], hue = 'TargetClass')
axes[0,0].set_xlabel('Apparent Magnitude')

sns.scatterplot(data = spectral_data, x = 'Amag',  y= 'B-V', ax = axes[0][1], hue = 'TargetClass')
axes[0,1].set_xlabel('Absolute Magnitude')


sns.scatterplot(data = spectral_class, x = 'Vmag', y = 'Plx', ax = axes[1][0], hue='TargetClass')
axes[1,0].set_xlabel('Apparent Magnitude')


sns.scatterplot(data = spectral_class, x = 'Amag', y = 'Plx', ax = axes[1][1], hue = 'TargetClass')
axes[1,1].set_xlabel('Absolute Magnitude')

# %%
Pair_plot1 = sns.pairplot(spec_data, hue = 'Spectral Type', diag_kind="hist")

Pair_plot1.axes[2,0].set_xlim((17,-17))
Pair_plot1.axes[2,2].set_ylim((17,-17))

Pair_plot1.savefig("pairplot.png")
plt.show()

# %%

# %%
spec_data["Spectral Type"].value_counts() #Unbalanced classes

# %%
spec_data.loc[spec_data['Spectral Type'].str.contains("Main Sequence") == True,
              "Spectral Type"] = "Dwarf Class"
spec_data.loc[spec_data['Spectral Type'].str.contains("Sub Dwarf") == True, "Spectral Type"] = "Dwarf Class"
spec_data.loc[spec_data['Spectral Type'].str.contains("White Dwarf") == True, "Spectral Type"] = "Dwarf Class"
spec_data.loc[spec_data['Spectral Type'].str.contains("Normal Giant") == True, "Spectral Type"] = "Giant Class"
spec_data.loc[spec_data['Spectral Type'].str.contains("Sub Giant") == True, "Spectral Type"] = "Giant Class"
spec_data.loc[spec_data['Spectral Type'].str.contains("Super Giant") == True, "Spectral Type"] = "Giant Class"
spec_data.loc[spec_data['Spectral Type'].str.contains("Bright Giant") == True, "Spectral Type"] = "Giant Class"

spec_data.head()

# %%
KDE_plt = sns.pairplot(spec_data, kind="kde")

KDE_plt.axes[2,0].set_xlim((17,-17))
KDE_plt.axes[2,2].set_ylim((17,-17))

plt.show()

# %%
plt.savefig("kde2.pdf", dpi=300)

# %%
sns.set(rc={"figure.figsize":(13,13), "axes.labelsize":12})

KDEplotmax= sns.kdeplot(data=spec_data, x='B-V Index', y='Absolute Magnitude', hue = "Spectral Type", fill = True)

KDEplotmax.set_ylim(17,-17)
KDEplotmax.set_xlim(-0.5,3.5)

plt.show()

# %%
sns.set(rc={"figure.figsize":(13,13), "axes.labelsize":12})

KDEplotmax= sns.kdeplot(data=spec_data, x='B-V Index', y='Absolute Magnitude', hue = "Spectral Type")

KDEplotmax.set_ylim(12,-10)
KDEplotmax.set_xlim(-0.5,2.0)
KDEplotmax.set(title = 'Kernel Density Estimate Plot of H-R Diagramme')
plt.show()
plt.savefig("KDEplotmax.pdf")

# %%
KDEplotmax.savefig("KDEplotmax1.png")

# %%
spec_data.head()

# %%
spec_data["Spectral Type"].value_counts()

# %%
sp_data = spec_data.copy()

# %%
sp_data.head()

# %%
sp_data_trans = sp_data.dtypes[sp_data.dtypes == np.object]  # filtering by string categoricals
sp_data_trans = sp_data_trans.index.tolist()  # list of categorical fields

sp_data[sp_data_trans].head().T

# %%
sp_data = pd.get_dummies(sp_data, columns=sp_data_trans, drop_first=True)
sp_data.describe().T

# %%
sp_data.head()

# %%
sp_data = sp_data.rename(columns={'Spectral Type_Giant Class':'Spectral Type' })

# %%
sp_data.head()

# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score


from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, jaccard_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# %%
#Splitting Data

X1 = sp_data.drop('Spectral Type', axis = 1, inplace = False)
Y = sp_data['Spectral Type']

X1.head()

# %%
Y.head()

# %%
X = X1.copy()

# %%
scaler = RobustScaler()
X = scaler.fit_transform(X)

# %%
print(X)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state = 0)

# %% [markdown]
# # Logistic Regression

# %%
# Model Logistic Regression

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
best_solver = ''
best_score_LR = 0
for i in solvers:
    LogReg = LogisticRegression(solver=i)
    LogReg.fit(X_train, Y_train)
    Y_pred_LR = LogReg.predict(X_test)
    score = LogReg.score(X_test,Y_test)
    
    if score > best_score_LR:
        best_score_LR = score
        best_solver = i
print('The best solver for the Logistic Regression is {}, with a {}% of accuracy in the test set.'.format(best_solver, best_score_LR*100))

# %%
# L2 penalty to shrink coefficients without removing any features from the model
penalty= 'l2'
# Our classification problem is binomial
multi_class = 'ovr'
# Use lbfgs for L2 penalty and multinomial classes
solver = 'saga'
# Max iteration = 1000
max_iter = 1000

# %%
l2_model = LogisticRegression(random_state=42, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

# %%
l2_model.fit(X_train, Y_train)

# %%
l2_preds = l2_model.predict(X_test)


# %%
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos


# %%
evaluate_metrics(Y_test, l2_preds)

# %%
print("Cross-Validation Score:", cross_val_score(l2_model, X_train, Y_train, cv=None, scoring="accuracy"))

# %%
print("Cross-Validation Score:", cross_val_score(l2_model, X_train, Y_train, cv=None, scoring=None))

# %%
target_names = ["0","1"]
y_cross_val_pred = cross_val_predict(l2_model,X_train,Y_train, cv=None)
print(classification_report(Y_train, y_cross_val_pred, target_names=target_names))

# %%
LR_matrix = plot_confusion_matrix(l2_model,X_test,Y_test,labels=[0,1], cmap='Reds')
plt.show()
plt.savefig("LR_matrix.png")

# %% [markdown]
# # SVC

# %%
kernels = ['rbf', 'poly', 'sigmoid','linear']
best_score_SVC = 0
kernel_best = ''
for i in kernels:
    SupVec = SVC(kernel=i, gamma='auto', random_state=1)
    SupVec.fit(X_train,Y_train)
    Y_pred_SV = SupVec.predict(X_test)
    if SupVec.score(X_test, Y_test) > best_score_SVC:
        best_score_SVC = SupVec.score(X_test, Y_test)
        kernel_best = i
    
print('The SVC performs better using a {} kernel, obtaining a {}% of accuracy'.format(kernel_best, best_score_SVC*100))
#print("The (test) accuracy is approximately {}%".format(round(SupVec.score(X_test,Y_test)*100)))

# %%
evaluate_metrics(Y_test, Y_pred_SV)

# %%
SVM_model = SVC(kernel="rbf", C=1, decision_function_shape="ovo")
SVM_model.fit(X_train, Y_train)

# %%
print("Cross-Validation Score:", cross_val_score(SVM_model, X_train, Y_train, cv=None, scoring=None))

# %%
y_SVM_pred = SVM_model.predict(X_test)
evaluate_metrics(Y_test, y_SVM_pred)

# %%
target_names = ["0","1"]
y_cross_val_pred_svm = cross_val_predict(SVM_model,X_train,Y_train, cv=None)
print(classification_report(Y_train, y_cross_val_pred_svm, target_names=target_names))

# %%
SVM_matrix = plot_confusion_matrix(SVM_model,X_test,Y_test,labels=[0,1], cmap='Greens')
plt.show()
plt.savefig("SVM_matrix.png")

# %% [markdown]
# # K-NN

# %%
n_neigh = 15
K_best = 0
Score_best = 0

for i in range(1,n_neigh):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(X_train,Y_train)
    Y_pred = KNN.predict(X_test)
    if KNN.score(X_test,Y_test) > Score_best:
        Score_best = KNN.score(X_test,Y_test)
        K_best = i
print("The best number of neighbors is {} with a test accuracy of {}%""".format(K_best, (Score_best*100)))



# %%
knn = KNeighborsClassifier(n_neighbors=14)
knn = knn.fit(X_train, Y_train)

# %%
y_pred_knn = knn.predict(X_test)
evaluate_metrics(Y_test, y_pred_knn)

# %%
print("Cross-Validation Score:", cross_val_score(SVM_model, X_train, Y_train, cv=None, scoring=None))

# %%
target_names = ["0","1"]
y_cross_val_pred_knn = cross_val_predict(knn,X_train,Y_train, cv=None)
print(classification_report(Y_train, y_cross_val_pred_knn, target_names=target_names))

# %%
knn_matrix = plot_confusion_matrix(knn,X_test,Y_test,labels=[0,1], cmap='Blues')
plt.show()
plt.savefig("knn_matrix.png")

# %%
