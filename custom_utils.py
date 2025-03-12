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
    df = df.drop(["WScore", "LScore", "NumOT", "WLoc"], axis=1)
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
    elif gender == "M" :
        df = df.loc[df.WTeamID > 3000]
    else:
        raise ValueError("You have to specify gender! Your options: W - women, M - men")
    X_features = df.iloc[:, 2:]
    pred_probs = np.max(clf.predict_proba(X_features), axis=1)
    df["Pred"] = np.round(pred_probs, 4)
    return df[["ID", "Pred"]]


def enrich_data(df: pd.DataFrame, gender: str):
    
    if gender == "W":
        seeds = pd.read_csv("data/WNCAATourneySeeds.csv")
    elif gender == "M":
        seeds = pd.read_csv("data/MNCAATourneySeeds.csv")
    else:
        raise ValueError("You have to specify gender! Your options: W - women, M - men")
    
    seeds["ISeed"] = seeds.apply(transform_seeds, axis=1)
    seeds["ISeed"] = seeds.apply(transform_seeds, axis=1)

    prep_enh = pd.merge(df, seeds, how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"]).drop(["TeamID", "Seed"], axis=1).fillna(16.)
    prep_enh["ISeed"] = prep_enh["ISeed"].astype("int")
    prep_enh = prep_enh.rename(columns={"ISeed" : "SeedW"})
    prep_enh = pd.merge(prep_enh, seeds, how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]).drop(["TeamID", "Seed"], axis=1).fillna(16.)
    prep_enh["ISeed"] = prep_enh["ISeed"].astype("int")
    prep_enh = prep_enh.rename(columns={"ISeed" : "SeedL"})
    prep_enh["SeedDiff"] = prep_enh["SeedW"] - prep_enh["SeedL"]
    return prep_enh