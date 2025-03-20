import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def transform_seeds(row):
    return int(row["Seed"][1:3])


def transform_loc(row):
    if row["WLoc"] == "A":
        return "H"
    elif row["WLoc"] == "H":
        return "A"
    else:
        return "N"


def get_enc_region(row):
    return row["Seed"][:1]


def get_real_region(row):
    region_map = {
        "W": row["RegionW"],
        "X": row["RegionX"],
        "Y": row["RegionY"],
        "Z": row["RegionZ"],
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
    df[["Season", "WTeamID", "LTeamID"]] = (
        df["ID"].str.split("_", expand=True).astype(int)
    )
    df["isTourney"] = 1
    if gender == "W":
        df = df.loc[df.WTeamID < 3000]
    elif gender == "M":
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
        mteams = pd.read_csv("data/MTeams.csv")
        mcoaches = pd.read_csv("data/MTeamCoaches.csv")
    else:
        raise ValueError("You have to specify gender! Your options: W - women, M - men")

    seeds["ISeed"] = seeds.apply(transform_seeds, axis=1)

    seeds_prev = seeds.copy()
    seeds_prev["Season"] += 1

    # information whether is tournament phase
    df["is_tournament"] = df["DayNum"] >= 136

    # add seed values based on tournament values
    regular_season = (
        df[~df["is_tournament"]]
        .merge(
            seeds_prev,
            how="left",
            left_on=["Season", "WTeamID"],
            right_on=["Season", "TeamID"],
        )
        .drop(["TeamID", "Seed"], axis=1)
    )
    regular_season = regular_season.rename(columns={"ISeed": "SeedW"})

    regular_season = regular_season.merge(
        seeds_prev,
        how="left",
        left_on=["Season", "LTeamID"],
        right_on=["Season", "TeamID"],
    ).drop(["TeamID", "Seed"], axis=1)
    regular_season = regular_season.rename(columns={"ISeed": "SeedL"})

    tournament = (
        df[df["is_tournament"]]
        .merge(
            seeds,
            how="left",
            left_on=["Season", "WTeamID"],
            right_on=["Season", "TeamID"],
        )
        .drop(["TeamID", "Seed"], axis=1)
    )
    tournament = tournament.rename(columns={"ISeed": "SeedW"})

    tournament = tournament.merge(
        seeds, how="left", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]
    ).drop(["TeamID", "Seed"], axis=1)
    tournament = tournament.rename(columns={"ISeed": "SeedL"})

    prep_enh = pd.concat([regular_season, tournament])

    # added coaches names, encode it and calculate how long team is in Division I
    if gender == "M":
        prep_enh = prep_enh.merge(
            mcoaches,
            how="left",
            left_on=["Season", "WTeamID"],
            right_on=["Season", "TeamID"],
        )
        prep_enh = (
            prep_enh.loc[
                (prep_enh.DayNum >= prep_enh.FirstDayNum)
                & (prep_enh.DayNum <= prep_enh.LastDayNum)
            ]
            .drop(["TeamID", "FirstDayNum", "LastDayNum"], axis=1)
            .rename(columns={"CoachName": "WCoachName"})
        )

        prep_enh = prep_enh.merge(
            mcoaches,
            how="left",
            left_on=["Season", "LTeamID"],
            right_on=["Season", "TeamID"],
        )
        prep_enh = (
            prep_enh.loc[
                (prep_enh.DayNum >= prep_enh.FirstDayNum)
                & (prep_enh.DayNum <= prep_enh.LastDayNum)
            ]
            .drop(["TeamID", "FirstDayNum", "LastDayNum"], axis=1)
            .rename(columns={"CoachName": "LCoachName"})
        )

        label_enc = LabelEncoder()
        prep_enh["WCoachName"] = label_enc.fit_transform(prep_enh[["WCoachName"]])
        prep_enh["LCoachName"] = label_enc.transform(prep_enh[["LCoachName"]])

        prep_enh = (
            prep_enh.merge(mteams, how="left", left_on="WTeamID", right_on="TeamID")
            .drop(["TeamID", "TeamName"], axis=1)
            .rename(columns={"Duration": "WDuration"})
        )
        prep_enh.loc[
            prep_enh["Season"] - prep_enh["FirstD1Season"]
            < prep_enh["LastD1Season"] - prep_enh["FirstD1Season"],
            "Duration",
        ] = (
            prep_enh["Season"] - prep_enh["FirstD1Season"] + 1
        )
        prep_enh.loc[
            prep_enh["Season"] - prep_enh["FirstD1Season"]
            >= prep_enh["LastD1Season"] - prep_enh["FirstD1Season"],
            "Duration",
        ] = (
            prep_enh["LastD1Season"] - prep_enh["FirstD1Season"] + 1
        )
        prep_enh = prep_enh.drop(["FirstD1Season", "LastD1Season"], axis=1)

    # change dtypes and calucalte difference between seeds
    prep_enh["SeedW"] = prep_enh["SeedW"].fillna(16).astype(int)
    prep_enh["SeedL"] = prep_enh["SeedL"].fillna(16).astype(int)
    prep_enh["is_tournament"] = prep_enh["is_tournament"].astype(int)

    prep_enh["SeedDiff"] = prep_enh["SeedW"] - prep_enh["SeedL"]

    return prep_enh


def highlight_top_n(s, n=5, color="green"):
    top_n = s.nlargest(n).values
    return [f"background-color: {color}" if v in top_n else "" for v in s]


def feature_importance(clf, data: pd.DataFrame) -> pd.DataFrame:
    importance_clf = clf.feature_importances_
    feature = clf.feature_names_in_

    importance_mi = mutual_info_classif(data.drop("Result", axis=1), data["Result"])

    print(importance_mi)

    importance_df = pd.DataFrame(
        {
            "Feature": feature,
            "Importance_clf": importance_clf,
            "Importance_mi": importance_mi,
        }
    )

    return importance_df.style.apply(
        highlight_top_n, subset=["Importance_clf", "Importance_mi"]
    )
