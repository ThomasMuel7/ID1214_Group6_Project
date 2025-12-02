import pandas as pd

# Load all data files
downs_defense = pd.read_excel('data/2025_downs_defense.xlsx')
downs_offense = pd.read_excel('data/2025_downs_offense.xlsx')
kickoff_offense = pd.read_excel('data/2025_kickoff-return_offense.xlsx')
kickoff_defense = pd.read_excel('data/2025_kickoff-return_defense.xlsx')
overall_defense = pd.read_excel('data/2025_overall_defense.xlsx')
overall_offense = pd.read_excel('data/2025_overall_offense.xlsx')
passing_defense = pd.read_excel('data/2025_passing_defense.xlsx')
passing_offense = pd.read_excel('data/2025_passing_offense.xlsx')
punt_defense = pd.read_excel('data/2025_punt-return_defense.xlsx')
punt_offense = pd.read_excel('data/2025_punt-return_offense.xlsx')
punting_defense = pd.read_excel('data/2025_punting_defense.xlsx')
punting_offense = pd.read_excel('data/2025_punting_offense.xlsx')
rushing_defense = pd.read_excel('data/2025_rushing_defense.xlsx')
rushing_offense = pd.read_excel('data/2025_rushing_offense.xlsx')
scoring_defense = pd.read_excel('data/2025_scoring_defense.xlsx')
scoring_offense = pd.read_excel('data/2025_scoring_offense.xlsx')
scores = pd.read_csv('data/2025_scores.csv')
upcoming_games = pd.read_excel('data/upcoming_games.xlsx')

# Drop 'gms' column from all datasets
datasets = [downs_defense, downs_offense, kickoff_offense, kickoff_defense,
            overall_defense, overall_offense, passing_defense, passing_offense,
            punt_defense, punt_offense, punting_defense, punting_offense,
            rushing_defense, rushing_offense, scoring_defense, scoring_offense]

for df in datasets:
    if 'Gms' in df.columns:
        df.drop('Gms', axis=1, inplace=True)

#combine all datasets into a single DataFrame with explicit suffixes
team_col = overall_offense.columns[0]
team_data = overall_offense.copy()
team_data.columns = [col if col == team_col else col + '_overall_offense' for col in team_data.columns]

team_data = team_data.merge(overall_defense.rename(columns={col: col + '_overall_defense' if col != team_col else col for col in overall_defense.columns}), on=team_col, how='outer')
team_data = team_data.merge(passing_offense.rename(columns={col: col + '_passing_offense' if col != team_col else col for col in passing_offense.columns}), on=team_col, how='outer')
team_data = team_data.merge(passing_defense.rename(columns={col: col + '_passing_defense' if col != team_col else col for col in passing_defense.columns}), on=team_col, how='outer')
team_data = team_data.merge(rushing_offense.rename(columns={col: col + '_rushing_offense' if col != team_col else col for col in rushing_offense.columns}), on=team_col, how='outer')
team_data = team_data.merge(rushing_defense.rename(columns={col: col + '_rushing_defense' if col != team_col else col for col in rushing_defense.columns}), on=team_col, how='outer')
team_data = team_data.merge(downs_offense.rename(columns={col: col + '_downs_offense' if col != team_col else col for col in downs_offense.columns}), on=team_col, how='outer')
team_data = team_data.merge(downs_defense.rename(columns={col: col + '_downs_defense' if col != team_col else col for col in downs_defense.columns}), on=team_col, how='outer')
team_data = team_data.merge(kickoff_offense.rename(columns={col: col + '_kickoff_offense' if col != team_col else col for col in kickoff_offense.columns}), on=team_col, how='outer')
team_data = team_data.merge(kickoff_defense.rename(columns={col: col + '_kickoff_defense' if col != team_col else col for col in kickoff_defense.columns}), on=team_col, how='outer')
team_data = team_data.merge(punt_offense.rename(columns={col: col + '_punt_offense' if col != team_col else col for col in punt_offense.columns}), on=team_col, how='outer')
team_data = team_data.merge(punt_defense.rename(columns={col: col + '_punt_defense' if col != team_col else col for col in punt_defense.columns}), on=team_col, how='outer')
team_data = team_data.merge(punting_offense.rename(columns={col: col + '_punting_offense' if col != team_col else col for col in punting_offense.columns}), on=team_col, how='outer')
team_data = team_data.merge(punting_defense.rename(columns={col: col + '_punting_defense' if col != team_col else col for col in punting_defense.columns}), on=team_col, how='outer')
team_data = team_data.merge(scoring_offense.rename(columns={col: col + '_scoring_offense' if col != team_col else col for col in scoring_offense.columns}), on=team_col, how='outer')
team_data = team_data.merge(scoring_defense.rename(columns={col: col + '_scoring_defense' if col != team_col else col for col in scoring_defense.columns}), on=team_col, how='outer')

# Add the data of each team to the scores
scores = scores.drop(columns='Overtime', errors='ignore')
new_scores = scores.copy()
team_data_home = team_data.copy()
team_data_visitor = team_data.copy()

# Rename columns to add _home and _visitor suffixes (except the team identifier column)
team_col = team_data.columns[0]
team_data_home = team_data_home.rename(columns={col: col + '_home' for col in team_data_home.columns if col != team_col})
team_data_visitor = team_data_visitor.rename(columns={col: col + '_visitor' for col in team_data_visitor.columns if col != team_col})

# Merge with home team data
new_scores = new_scores.merge(team_data_home, left_on='Home', right_on=team_col, how='left')
upcoming_games = upcoming_games.merge(team_data_home, left_on='Home', right_on=team_col, how='left')
# Merge with visitor team data
new_scores = new_scores.merge(team_data_visitor, left_on='Visitor', right_on=team_col, how='left')
upcoming_games = upcoming_games.merge(team_data_visitor, left_on='Visitor', right_on=team_col, how='left')

# Drop the team identifier columns that were created during merge
new_scores = new_scores.drop(columns=[team_col + '_x', team_col + '_y'], errors='ignore')
upcoming_games = upcoming_games.drop(columns=[team_col + '_x', team_col + '_y'], errors='ignore')

# Calculate differences (home - visitor) for all team statistics
home_cols = [col for col in new_scores.columns if col.endswith('_home')]
for col in home_cols:
    visitor_col = col.replace('_home', '_visitor')
    if visitor_col in new_scores.columns:
        diff_col = col.replace('_home', '_diff')
        new_scores[diff_col] = new_scores[col] - new_scores[visitor_col]

upcoming_cols = [col for col in upcoming_games.columns if col.endswith('_home')]
for col in upcoming_cols:
    visitor_col = col.replace('_home', '_visitor')
    if visitor_col in upcoming_games.columns:
        diff_col = col.replace('_home', '_diff')
        upcoming_games[diff_col] = upcoming_games[col] - upcoming_games[visitor_col]
        
# Drop the _home and _visitor columns to keep only differences
new_scores = new_scores.drop(columns=[col for col in new_scores.columns if col.endswith('_home') or col.endswith('_visitor')], errors='ignore')
new_scores['Home_win'] = (new_scores['Home_pts'] > new_scores['Visitor_pts']).astype(int)
new_scores = new_scores.drop(columns=['Home_pts', 'Visitor_pts'], errors='ignore')

upcoming_games = upcoming_games.drop(columns=[col for col in upcoming_games.columns if col.endswith('_home') or col.endswith('_visitor')], errors='ignore')
upcoming_games = upcoming_games.drop(columns=['Home_pts', 'Visitor_pts'], errors='ignore')
# Save the final dataset
new_scores.to_csv('data/2025_processed_scores.csv', index=False)
upcoming_games.to_csv('data/2025_processed_upcoming_games.csv', index=False)