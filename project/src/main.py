import pandas as pd
from dataloader import DataLoader
from preprocessing import Preprocessor
from preprocessing import splitdata
from pcavisualizer import PCAVisualizer
from classificationmodels import Model
from classificationmodels import bagging_decisiontree, boosting_decisiontree, bagging_knn, bagging_svc


# LOAD DATASETS
loader = DataLoader(
    patients_path="../data/patients.txt",
    fitbit_path="../data/fitbit_daily.txt"
)

loader.load()

loader.printhead(loader.patients_path).printhead(loader.fitbit_path)
loader.printinfo(loader.patients_path).printinfo(loader.fitbit_path)

# CLEANING, MERGING AND SPLITTING DATASETS
df_pat = loader.get_patients()
df_fit = loader.get_fitbit()

col_to_drop_pat = [
    "ObjectId", "Centro", "Sede",
    "PatologiaPrincipale", "Gruppo"
]
col_to_drop_fit = [
    "TimeInBed", "AsleepMinutes",
    "LightSleepCounts", "DeepSleepCounts", "REMSleepCounts",
    "AwakeCounts", "AsleepCounts", "LightSleepBreathingRate",
    "DeepSleepBreathingRate", "REMSleepBreathingRate"
]
zero_cols_fit = [
    "MinutesAfterWakeup", "MinutesToFallAsleep", "SleepEfficiency",
    "LightSleepMinutes", "DeepSleepMinutes", "REMSleepMinutes",
    "AwakeMinutes"
]
median_cols_fit = [
     "Calories", "ActivityCalories", "RestingHeartRate",
    "VO2Max", "NightlySkinTemperature", "BreathingRate",
    "FullSleepBreathingRate",
    "BelowFatBurnCalories", "FatBurnCalories", "CardioCalories", "PeakCalories",
    "BelowFatBurnMinutes", "FatBurnMinutes", "CardioMinutes", "PeakMinutes"
]

preproc = Preprocessor(df_pat, df_fit)

preproc.check_duplicates(df_pat, "patients") \
       .drop_cols(df_pat, col_to_drop_pat) \
       .encode(df_pat, "Sesso", {'female': 0, 'male': 1}) \
       .datetoage(df_pat, "DataNascita")

preproc.check_duplicates(df_fit, "fitbit") \
       .drop_cols(df_fit, col_to_drop_fit) \
       .impute(df_fit, zero_cols_fit, "zero") \
       .impute(df_fit, median_cols_fit, "median")

preproc.drop_unmatched_patients()
df_final = preproc.merge()

print(df_final.head())
print(df_final.info())

X = df_final.drop(columns=["PatientID", "Day","GruppoLabel"])
y = df_final["GruppoLabel"]
X_train, X_test, y_train, y_test = splitdata(X, y)

# VISUALIZE PCA SCATTERPLOT AND VARIANCE RATIO
pca = PCAVisualizer(2)
Xpca = pca.reduce(X)
pca.plot(Xpca,y)
print(f"Explained variance ratio: {pca.explainvariance()}")

# TRAINING CLASSIFICATION MODELS AND VALIDATION
Xtrain, Xtest, ytrain, ytest = splitdata(X=X,y=y)

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