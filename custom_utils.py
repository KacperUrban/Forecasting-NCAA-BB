import fireducks.pandas as pd
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
        regularseason = pd.read_csv("data/WRegularSeasonCompactResults.csv")
        tourneyseason = pd.read_csv("data/WNCAATourneyCompactResults.csv")
    elif gender == "M":
        seeds = pd.read_csv("data/MNCAATourneySeeds.csv")
        mteams = pd.read_csv("data/MTeams.csv")
        mcoaches = pd.read_csv("data/MTeamCoaches.csv")
        regularseason = pd.read_csv("data/MRegularSeasonCompactResults.csv")
        tourneyseason = pd.read_csv("data/MNCAATourneyCompactResults.csv")
        mrankings = pd.read_csv("data/MMasseyOrdinals.csv")
        mrankings = mrankings[mrankings.Season > 2015].reset_index(drop=True)
        medianrankings = mrankings.groupby(["Season", "RankingDayNum", "TeamID"]).agg({"OrdinalRank": "median"}).reset_index()
    else:
        raise ValueError("You have to specify gender! Your options: W - women, M - men")

    # concat basic data for further improvement
    data = pd.concat([regularseason, tourneyseason], axis=0)
    data = data.loc[data.Season > 2015].reset_index(drop=True)
    data['Win'] = 1


    prep_wins = df.merge(
        data,
        how="left",
        on=["Season", "WTeamID"],
    ).drop(["WLoc", "NumOT"], axis=1)

    # add sum of wins (calculate every win till DayNum)
    wins = prep_wins[prep_wins["DayNum_x"] - prep_wins["DayNum_y"] > 0 ].reset_index(drop=True)
    wins = wins.groupby(["Season", "DayNum_x", "WTeamID", "LTeamID_x"]).agg({"Win" : "sum"}).reset_index().rename(columns={"DayNum_x" : "DayNum", "LTeamID_x" : "LTeamID",
                                                                                                                    "Win" : "Wwin"})

    prep_wins = df.merge(
        data,
        how="left",
        on=["Season", "LTeamID"],
    ).drop(["WLoc", "NumOT"], axis=1)

    lwins = prep_wins[prep_wins["DayNum_x"] - prep_wins["DayNum_y"] > 0 ].reset_index(drop=True)
    lwins = lwins.groupby(["Season", "DayNum_x", "WTeamID_x", "LTeamID"]).agg({"Win" : "sum"}).reset_index().rename(columns={"DayNum_x" : "DayNum", "WTeamID_x" : "WTeamID",
                                                                                                                    "Win" : "Lwin"})
    complete_wins = lwins.merge(
        wins,
        on=["Season", "DayNum", "WTeamID", "LTeamID"])
    
    #add mean score from all previous games in particular Season
    onehalf_result = data[["Season", "DayNum", "WTeamID", "WScore"]].rename(columns={"WTeamID" : "TeamID", "WScore" : "Score"})
    secondhalf_result = data[["Season", "DayNum", "LTeamID", "LScore"]].rename(columns={"LTeamID" : "TeamID", "LScore" : "Score"})
    wfull_result = pd.concat([onehalf_result, secondhalf_result], axis=0).reset_index(drop=True)

    lscores = complete_wins.merge(
        wfull_result,
        how="left",
        left_on=["Season", "WTeamID"],
        right_on=["Season", "TeamID"]
    )
    lscores = lscores[lscores.DayNum_x - lscores.DayNum_y > 0].reset_index(drop=True)
    lscores = lscores.groupby(["Season", "DayNum_x", "WTeamID", "LTeamID"]).agg({"Score" : "mean"}).reset_index().rename(columns={"DayNum_x" : "DayNum", 
                                                                                                                                "Score" : "WScore"})

    rscores = complete_wins.merge(
        wfull_result,
        how="left",
        left_on=["Season", "LTeamID"],
        right_on=["Season", "TeamID"]
    )
    rscores = rscores[rscores.DayNum_x - rscores.DayNum_y > 0].reset_index(drop=True)
    rscores = rscores.groupby(["Season", "DayNum_x", "WTeamID", "LTeamID"]).agg({"Score" : "mean"}).reset_index().rename(columns={"DayNum_x" : "DayNum", 
                                                                                                                                "Score" : "LScore"})
    complete_scores = rscores.merge(
        lscores,
        how="left",
        on=["Season", "DayNum", "WTeamID", "LTeamID"]
    )
    complete_wins_scores = complete_scores.merge(
        complete_wins,
        how="left",
        on=["Season", "DayNum", "WTeamID", "LTeamID"]
    )
    
    df = df.merge(
        complete_wins_scores,
        how="left",
        on=["Season", "DayNum", "WTeamID", "LTeamID"]
    ).fillna(0)

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

    # add coaches names, encode it and calculate how long team is in Division I
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

        prep_enh_rank = prep_enh.merge(
            medianrankings,
            how="left",
            left_on=["Season", "WTeamID"], 
            right_on=["Season", "TeamID"]
        )

        prep_enh_rank = prep_enh_rank[prep_enh_rank.RankingDayNum - prep_enh_rank.DayNum <= 0].reset_index(drop=True)
        prep_enh_rank = prep_enh_rank.groupby(["Season", "DayNum", "WTeamID", "LTeamID"]).agg(lambda x: x.iloc[-1]).reset_index()\
            .drop(["RankingDayNum", "TeamID"], axis=1).rename(columns={"OrdinalRank": "RankW"})

        prep_enh_rank = prep_enh_rank.merge(
            medianrankings,
            how="left",
            left_on=["Season", "LTeamID"], 
            right_on=["Season", "TeamID"]
        )

        prep_enh_rank = prep_enh_rank[prep_enh_rank.RankingDayNum - prep_enh_rank.DayNum <= 0].reset_index(drop=True)
        prep_enh_rank = prep_enh_rank.groupby(["Season", "DayNum", "WTeamID", "LTeamID"]).agg(lambda x: x.iloc[-1]).reset_index()\
            .drop(["RankingDayNum", "TeamID"], axis=1).rename(columns={"OrdinalRank": "RankL"})
        prep_enh = prep_enh_rank

    # change dtypes and calculate difference between seeds
    prep_enh["SeedW"] = prep_enh["SeedW"].fillna(16).astype(int)
    prep_enh["SeedL"] = prep_enh["SeedL"].fillna(16).astype(int)
    prep_enh["is_tournament"] = prep_enh["is_tournament"].astype(int)

    prep_enh["SeedDiff"] = prep_enh["SeedW"] - prep_enh["SeedL"]

    return prep_enh.reset_index(drop=True)


def highlight_top_n(s, n=5, color="green"):
    top_n = s.nlargest(n).values
    return [f"background-color: {color}" if v in top_n else "" for v in s]


def feature_importance(clf, data: pd.DataFrame) -> pd.DataFrame:
    importance_clf = clf.feature_importances_
    feature = clf.feature_names_in_

    importance_mi = mutual_info_classif(data.drop("Result", axis=1), data["Result"])

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
