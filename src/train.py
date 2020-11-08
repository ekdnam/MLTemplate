from sklearn import preprocessing
import pandas as pd
import os
from sklearn import ensemble
from sklearn import metrics
import joblib
# from . import dispatcher
import dispatcher
# TRAINING_DATA = os.environ.get("TRAINING_DATA")
# Fold = os.environ.get("FOLD")

TRAINING_DATA = "input/train_folds.csv"
FOLD = 0
# FOLD = 1
# FOLD = 2
# FOLD = 3
# FOLD = 4

MODEL = 'extratrees'

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [1, 0, 3, 4],
    3: [1, 2, 0, 4],
    4: [1, 2, 3, 0],
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c, lbl))

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    # print(probs)
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoders, f"models/{MODEL}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}.pkl")