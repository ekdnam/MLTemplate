import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    df["kfold"] = -1

    # df.sample returns a random sample from an axis of object
    # Reset the index, or a level of it.
    df = df.sample(frac=1).reset_index(drop=True)

    # Stratified K-Folds cross-validator
    # Provides train/test indices to split data in train/test sets.
    # This cross-validation object is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class.
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=12)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, "kfold"] = fold

    df.to_csv("input/train_folds.csv", index=False)
