import os
import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "data"
SCORES_DIR = os.path.join(DATA_DIR, "scores")

def parse_html(box_score):
    with open(box_score) as f:
        html = f.read()

    soup = BeautifulSoup(html)
    [s.decompose() for s in soup.select("tr.over_header")] # Remove headers from HTML
    [s.decompose() for s in soup.select("tr.thead")]
    return soup

def read_line_score(soup):
    line_score = pd.read_html(str(soup), attrs = {'id': 'line_score'})[0] # Read table with id 'line_score'
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    line_score = line_score[["team", "total"]]
    return line_score

def read_stats(soup, team, stat_type):
    df = pd.read_html(str(soup), attrs = {'id': f'box-{team}-game-{stat_type}'}, index_col=0)[0] # Read boxscore into df (advanced or basic stats determined by 'stat')
    df = df.apply(pd.to_numeric, errors="coerce") # Convert box score values to numeric
    return df

# Find season from boxscore
def read_season_info(soup):
    # /leagues/NBA_2016_games.html => get the '2016' from link of this form
    nav = soup.select("#bottom_nav_container")[0] # id = bottom_nav_container
    hrefs = [a["href"] for a in nav.find_all('a')] # Get all links
    season = os.path.basename(hrefs[1]).split("_")[0] # Split on underscore
    return season

box_scores = os.listdir(SCORES_DIR)
box_scores = [os.path.join(SCORES_DIR, f) for f in box_scores if f.endswith(".html")]

games = []
base_cols = None
for box_score in box_scores:
    soup = parse_html(box_score)

    line_score = read_line_score(soup)
    teams = list(line_score["team"])

    summaries = []
    for team in teams:
        basic = read_stats(soup, team, "basic")
        advanced = read_stats(soup, team, "advanced")

        totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]]) # 'totals' line of boxscore from last row
        totals.index = totals.index.str.lower() # Col names to lowercase

        maxes = pd.concat([basic.iloc[:-1].max(), advanced.iloc[:-1].max()]) # Get max values for each column (exclude last row aka totals)
        maxes.index = maxes.index.str.lower() + "_max" # Col names to lowercase and add '_max' to differentiate from totals

        summary = pd.concat([totals, maxes])
        
        # Some boxscores have different columns - take cols from first boxscore processed and use for all future boxscores
        if base_cols is None:
            base_cols = list(summary.index.drop_duplicates(keep="first"))
            base_cols = [b for b in base_cols if "bpm" not in b]
        
        summary = summary[base_cols]
        
        summaries.append(summary)
        
    summary = pd.concat(summaries, axis=1).T # Rotate df

    game = pd.concat([summary, line_score], axis=1) # Add line score 

    game["home"] = [0,1] # 0 if away, 1 if home

    # Restructure df - instead of first row for away team stats and second row for home team stats, create cols that indicate home/away team stats
    game_opp = game.iloc[::-1].reset_index()
    game_opp.columns += "_opp"
    full_game = pd.concat([game, game_opp], axis=1)

    full_game["season"] = read_season_info(soup)
    full_game["date"] = os.path.basename(box_score)[:8] # Date from first 8 characters of file name
    full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")
    
    full_game["won"] = full_game["total"] > full_game["total_opp"]
    games.append(full_game)
    
    games_df = pd.concat(games, ignore_index=True)
    games_df.to_csv("nba_games1.csv")
