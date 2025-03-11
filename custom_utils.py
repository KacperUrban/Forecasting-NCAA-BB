import pandas as pd
import numpy as np

def transform_seeds(row):
    return int(row["Seed"][1:3])

def transform_loc(row):
    if row["WLoc"] == "A":
        return 'H'
    elif row["WLoc"] == "H":
        return 'A'
    else:
        return 'N'

def get_enc_region(row):
    return row["Seed"][:1]

def get_real_region(row):
    region_map = {
        "W": row["RegionW"],
        "X": row["RegionX"],
        "Y": row["RegionY"],
        "Z": row["RegionZ"]
    }
    return region_map.get(row["RegionEnc"], "Unknown")

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["DayNum", "WScore", "LScore", "NumOT", "WLoc"], axis=1)
    df["Result"] = 1
    inv_df = df.copy()
    inv_df[["WTeamID", "LTeamID"]] = inv_df[["LTeamID", "WTeamID"]]
    inv_df["Result"] = 0
    
    merged_df = pd.concat([df, inv_df], axis=0).reset_index(drop=True)
    return merged_df

def make_preds_for_submission(clf, filepath_sub: str, gender: str) -> pd.DataFrame:
    df = pd.read_csv(filepath_sub)
    df[['Season', 'WTeamID', 'LTeamID']] = df['ID'].str.split('_', expand=True).astype(int)
    df['isTourney'] = 1
    if gender == "W":
        df = df.loc[df.WTeamID < 3000]
    else:
        df = df.loc[df.WTeamID > 3000]
    X_features = df.iloc[:, 2:]
    pred_probs = np.max(clf.predict_proba(X_features), axis=1)
    df["Pred"] = np.round(pred_probs, 4)
    return df[["ID", "Pred"]]