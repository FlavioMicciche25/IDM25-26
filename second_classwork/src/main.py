import pandas as pd
from dataloader import DataLoader
from preprocessing import Preprocessor
from pcavisualizer import PCAVisualizer
from preprocessing import splitdata
from classification_models import Model
from classification_models import bagging_decisiontree, boosting_decisiontree, bagging_knn, bagging_svc

loader = DataLoader("./Dataset DAES.xlsx")
loader.load()

datasets = {}
dfnames = ["ASD","GDD","Controlli"]

#LOAD DATASET
for name in dfnames:
    print(f"======{name}======")
    datasets[name] = loader.get_sheet(name)
    loader.printfirstrow(name,10)
    loader.printinfo(name)


#CLEANING DATASETS
coltorm = ["Pazienti","Età cronologica (mesi)","Scala B","Scala D","TOT.","Score di rischio"]
for name in dfnames:
    preproc = Preprocessor(datasets[name])
    datasets[name] = preproc.rmrows()
    datasets[name] = preproc.rmcolumns(cols=coltorm)
    datasets[name] = preproc.encode(["Sesso"])


#PREPARING FOR CLASSIFICATION
dflist = []
for name, df in datasets.items():
    dfc = df.copy()
    dfc["class"] = name
    dflist.append(dfc)

dfconcatenated = pd.concat(dflist, ignore_index=True)

dfconcatenated = dfconcatenated.dropna()

X = dfconcatenated.drop(columns=['class'])
y = dfconcatenated['class']

pca = PCAVisualizer(2)
Xpca = pca.reduce(X)
pca.plot(Xpca,y)
print(f"Explained variance ratio: {pca.explainvariance()}")

Xtrain, Xtest, ytrain, ytest = splitdata(X=X,y=y)

#CLASSIFICATION AND MODELS VALIDATION
models = ["decisiontree", "randomforest", "svc", "knn"]
result = {}
print()

for name in models:
    trainer = Model(name)
    trainer.train(Xtrain,ytrain)


    accuracy = trainer.evaluate(Xtest,ytest)
    result[name] = accuracy

    print(f"{name}")
    print("Best params:", trainer.bestparams())
    print("Test accuracy:", accuracy)
    print()

bagg_accuracy = bagging_decisiontree(Xtrain, ytrain, Xtest, ytest)
print(f"Bagging Decision Tree accuracy: {bagg_accuracy}")

boost_accuracy = boosting_decisiontree(Xtrain, ytrain, Xtest, ytest)
print(f"Boosting Decision Tree accuracy: {boost_accuracy}")

bagg_accuracy = bagging_knn(Xtrain, ytrain, Xtest, ytest)
print(f"Bagging K-NN accuracy: {bagg_accuracy}")

bagg_accuracy = bagging_svc(Xtrain, ytrain, Xtest, ytest)
print(f"Bagging SVC accuracy: {bagg_accuracy}")