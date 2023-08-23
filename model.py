import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier


def get_data():
    return pd.read_csv('nba_games.csv')

def clean_data(games):
    games = games.sort_values("date")
    games = games.reset_index(drop=True)
    
    # Group by team -> create a target feature which contains the result of the next game
    def createTarget(group):
        group["target"] = group["won"].shift(-1)
        return group
    games = games.groupby("team", group_keys=False).apply(createTarget)
    # The last game of a season will have no 'next game'. Set these games 'target' to 2
    games["target"][pd.isnull(games["target"])] = 2
    # Convert target to integer (0,1,2)
    games["target"] = games["target"].astype(int, errors="ignore")

    # Repeated columns and null columns
    toDelete = ["mp.1", "mp_opp.1", "index_opp", "Unnamed: 0"]
    toKeep = games.columns[~games.columns.isin(toDelete)]
    games = games[toKeep]
    nulls = pd.isnull(games).sum()
    nulls = nulls[nulls > 0]
    valid_columns = games.columns[~games.columns.isin(nulls.index)]
    games = games[valid_columns]

    return games

def make_predictions(data, model, predictors, start=2, step=1):
    all_predictions = []
    
    seasons = sorted(data["season"].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)

    return pd.concat(all_predictions)

def find_team_averages(team, num_games_averaged):
        rolling = team.rolling(num_games_averaged).mean()
        return rolling

# Group by team, create a new column
def shift_col(team, col_name):
        next_col = team[col_name].shift(-1)
        return next_col

def add_col_shifted(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

def find_matchup_wr(games_opp_next, games):
    team, season, opp = games_opp_next.iloc[0]["team"], games_opp_next.iloc[0]["season"], games_opp_next.iloc[0]["team_opp_next"]
    games = games[(games["team"] == team) & (games["season"] == season) & (games["team_opp"] == opp)] 

    winRates = [0.5]
    for i in range(1, len(games_opp_next)):
        prior_games = games[games["date"] <= games_opp_next.iloc[i]["date"]]
        if prior_games.empty:
             winRates.append(winRates[i-1])
             continue
        numWins = len(prior_games[prior_games["won"] == True])
        winRates.append(numWins/len(prior_games))

    games_opp_next["wr_vs_opp_next"] = winRates
    return games_opp_next

def create_model(games):
    # Things to split data, create model, select good features
    split = TimeSeriesSplit(n_splits=3)
    # model = GradientBoostingClassifier(n_estimators=55, 
    #                                    max_depth=3, 
    #                                    min_samples_leaf=10)
    model = RidgeClassifier(alpha=1)
    sfs = SequentialFeatureSelector(model, 
                                    n_features_to_select=60, 
                                    direction="backward",
                                    cv=split,
                                    n_jobs=1)
    
    # Scale features (to values between 0 and 1)
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    selected_columns = games.columns[~games.columns.isin(removed_columns)]
    scaler = MinMaxScaler()
    games[selected_columns] = scaler.fit_transform(games[selected_columns])

    # Create columns containing rolling averages for all numerical stats - consider a team's stats across their past 10 games to predict win/loss
    games_rolling = games[list(selected_columns) + ["won", "team", "season"]]
    num_games_averaged = 15
    games_rolling = games_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages, num_games_averaged)
    rolling_cols = [f"{col}_avg" for col in games_rolling.columns]
    games_rolling.columns = rolling_cols
    games = pd.concat([games, games_rolling], axis=1)
    games = games.dropna()

    # Create columns to account for home games, next opponent, and date of next game
    games["home_next"] = games.groupby("team", group_keys=False)["home"].shift(-1)
    games["team_opp_next"] = games.groupby("team", group_keys=False)["team_opp"].shift(-1)
    games["date_next"] = games.groupby("team", group_keys=False)["date"].shift(-1)
    games = games.dropna()

    # Create column containing winrate vs next opponent over season
    games = games.groupby(["team", "season", "team_opp_next"]).apply(find_matchup_wr, games)

    full = games.merge(games[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

    # Get selected features 
    # removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns
    # selected_columns = full.columns[~full.columns.isin(removed_columns)]
    # sfs.fit(full[selected_columns], full["target"])
    # predictors = list(selected_columns[sfs.get_support()])
    # print(predictors)

    print("-----------------------------------------------------------------------------------------------------------------------------------")
    
    predictors = ['won_avg_x', 'stl%', 'fg_max', 'pf_max', '+/-_max', 'ast%_max', 'tov%_max', 'total', 'ft%_opp', 'drb_opp', 'blk_opp', 'ast%_opp', 'blk%_opp', 'drb_max_opp', 'ast_max_opp', 'trb_avg_x', 'ast_avg_x', '3par_avg_x', 'ft_max_avg_x', 'orb_max_avg_x', 'pf_max_avg_x', 'pts_max_avg_x', 'ts%_max_avg_x', 'trb%_max_avg_x', 'fg_opp_avg_x', '3pa_opp_avg_x', 'stl_opp_avg_x', 'blk_opp_avg_x', '3p%_max_opp_avg_x', 'stl_max_opp_avg_x', 'pf_max_opp_avg_x', '+/-_max_opp_avg_x', 'usg%_max_opp_avg_x', 'drtg_max_opp_avg_x', 'home_opp_avg_x', 'home_next', 'fg%_avg_y', 'ft%_avg_y', 'ast_avg_y', '3par_avg_y', 'tov%_avg_y', '3p_max_avg_y', 'ft_max_avg_y', 'tov_max_avg_y', 'pts_max_avg_y', '3par_max_avg_y', 'orb%_max_avg_y', 'fg%_opp_avg_y', '3p%_opp_avg_y', 'drb_opp_avg_y', 'pf_opp_avg_y', '3par_opp_avg_y', 'blk%_opp_avg_y', '3p_max_opp_avg_y', '3p%_max_opp_avg_y', 'orb_max_opp_avg_y', '+/-_max_opp_avg_y', 'drb%_max_opp_avg_y', 'total_opp_avg_y', 'home_opp_avg_y', 'won_avg_y']
    predictions = make_predictions(full, model, predictors)
    predictions = predictions[predictions["actual"] != 2]
    print(accuracy_score(predictions["actual"], predictions["prediction"]))
    return 1


games = get_data()
gamesCleaned = clean_data(games)
model = create_model(gamesCleaned)
