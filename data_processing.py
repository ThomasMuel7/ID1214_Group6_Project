import pandas as pd
import os

# Create output directory
os.makedirs('processed_data', exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

# Stats
downs_defense = pd.read_excel('prep_data/2025_downs_defense.xlsx')
downs_offense = pd.read_excel('prep_data/2025_downs_offense.xlsx')
kickoff_offense = pd.read_excel('prep_data/2025_kickoff-returns_offense.xlsx')
kickoff_defense = pd.read_excel('prep_data/2025_kickoff-returns_defense.xlsx')
overall_defense = pd.read_excel('prep_data/2025_totals_defense.xlsx')
overall_offense = pd.read_excel('prep_data/2025_totals_offense.xlsx')
passing_defense = pd.read_excel('prep_data/2025_passing_defense.xlsx')
passing_offense = pd.read_excel('prep_data/2025_passing_offense.xlsx')
punt_defense = pd.read_excel('prep_data/2025_punt-returns_defense.xlsx')
punt_offense = pd.read_excel('prep_data/2025_punt-returns_offense.xlsx')
punting_defense = pd.read_excel('prep_data/2025_punting_defense.xlsx')
punting_offense = pd.read_excel('prep_data/2025_punting_offense.xlsx')
rushing_defense = pd.read_excel('prep_data/2025_rushing_defense.xlsx')
rushing_offense = pd.read_excel('prep_data/2025_rushing_offense.xlsx')
scoring_defense = pd.read_excel('prep_data/2025_scoring_defense.xlsx')
scoring_offense = pd.read_excel('prep_data/2025_scoring_offense.xlsx')

# Scores and Schedules
scores = pd.read_excel('prep_data/2025_scores.xlsx')
upcoming_games = pd.read_excel('prep_data/2025_upcoming_games.xlsx')

# ---------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------

datasets = [downs_defense, downs_offense, kickoff_offense, kickoff_defense,
            overall_defense, overall_offense, passing_defense, passing_offense,
            punt_defense, punt_offense, punting_defense, punting_offense,
            rushing_defense, rushing_offense, scoring_defense, scoring_offense]

for df in datasets:
    # Drop 'Gms' or 'Games' column if it exists
    cols_to_drop = [c for c in df.columns if 'Gms' in str(c) or 'Games' in str(c)]
    if cols_to_drop:
        df.drop(cols_to_drop, axis=1, inplace=True)
    
    team_col_name = df.columns[0]
    df[team_col_name] = df[team_col_name].astype(str).str.strip()

# ---------------------------------------------------------
# 3. MERGING STATS
# ---------------------------------------------------------

# Initialize team_data with overall_offense
# We assume the first column is the Team Name
team_col = overall_offense.columns[0]
team_data = overall_offense.copy()
# Rename columns to include suffix
team_data.columns = [col if col == team_col else col + '_overall_offense' for col in team_data.columns]

# Helper function to merge datasets
def merge_stat(base_df, new_df, suffix):
    # The joining column is assumed to be the first column in both
    join_col_base = base_df.columns[0]
    join_col_new = new_df.columns[0]
    
    # Rename columns in the new dataframe
    renamed_df = new_df.rename(columns={
        col: col + suffix if col != join_col_new else col 
        for col in new_df.columns
    })
    
    # Merge on the team name column
    return base_df.merge(renamed_df, left_on=join_col_base, right_on=renamed_df.columns[0], how='outer')

# Merge all statistics into one 'team_data' dataframe
team_data = merge_stat(team_data, overall_defense, '_overall_defense')
team_data = merge_stat(team_data, passing_offense, '_passing_offense')
team_data = merge_stat(team_data, passing_defense, '_passing_defense')
team_data = merge_stat(team_data, rushing_offense, '_rushing_offense')
team_data = merge_stat(team_data, rushing_defense, '_rushing_defense')
team_data = merge_stat(team_data, downs_offense, '_downs_offense')
team_data = merge_stat(team_data, downs_defense, '_downs_defense')
team_data = merge_stat(team_data, kickoff_offense, '_kickoff_offense')
team_data = merge_stat(team_data, kickoff_defense, '_kickoff_defense')
team_data = merge_stat(team_data, punt_offense, '_punt_offense')
team_data = merge_stat(team_data, punt_defense, '_punt_defense')
team_data = merge_stat(team_data, punting_offense, '_punting_offense')
team_data = merge_stat(team_data, punting_defense, '_punting_defense')
team_data = merge_stat(team_data, scoring_offense, '_scoring_offense')
team_data = merge_stat(team_data, scoring_defense, '_scoring_defense')

# ---------------------------------------------------------
# 4. MERGE INTO SCORES AND UPCOMING
# ---------------------------------------------------------

scores = scores.drop(columns=['Overtime', 'OT'], errors='ignore')
new_scores = scores.copy()
upcoming_processed = upcoming_games.copy()

team_data_home = team_data.copy()
team_data_visitor = team_data.copy()

# The unique key in team_data is the first column (Team Name)
join_key = team_data.columns[0]

# Add _home / _visitor suffixes
team_data_home = team_data_home.rename(columns={col: col + '_home' for col in team_data_home.columns if col != join_key})
team_data_visitor = team_data_visitor.rename(columns={col: col + '_visitor' for col in team_data_visitor.columns if col != join_key})

# FIX: Clean Team Names in the Score files too
new_scores['Home'] = new_scores['Home'].astype(str).str.strip()
new_scores['Visitor'] = new_scores['Visitor'].astype(str).str.strip()
upcoming_processed['Home'] = upcoming_processed['Home'].astype(str).str.strip()
upcoming_processed['Visitor'] = upcoming_processed['Visitor'].astype(str).str.strip()

# Merge Home Stats
new_scores = new_scores.merge(team_data_home, left_on='Home', right_on=join_key, how='left')
upcoming_processed = upcoming_processed.merge(team_data_home, left_on='Home', right_on=join_key, how='left')

# Merge Visitor Stats
new_scores = new_scores.merge(team_data_visitor, left_on='Visitor', right_on=join_key, how='left')
upcoming_processed = upcoming_processed.merge(team_data_visitor, left_on='Visitor', right_on=join_key, how='left')

# Drop the extra team name columns created by the merge
cols_to_drop = [join_key + '_x', join_key + '_y']
new_scores = new_scores.drop(columns=cols_to_drop, errors='ignore')
upcoming_processed = upcoming_processed.drop(columns=cols_to_drop, errors='ignore')

# ---------------------------------------------------------
# 5. CALCULATE DIFFERENTIALS
# ---------------------------------------------------------

def calculate_diffs(df):
    home_cols = [col for col in df.columns if col.endswith('_home')]
    for col in home_cols:
        visitor_col = col.replace('_home', '_visitor')
        if visitor_col in df.columns:
            diff_col = col.replace('_home', '_diff')
            # Coerce to numeric ensures that even if Excel read some numbers as strings, the math works
            df[diff_col] = pd.to_numeric(df[col], errors='coerce') - pd.to_numeric(df[visitor_col], errors='coerce')
    
    # Remove the raw _home and _visitor columns
    cols_to_remove = [c for c in df.columns if c.endswith('_home') or c.endswith('_visitor')]
    df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    return df

new_scores = calculate_diffs(new_scores)
upcoming_processed = calculate_diffs(upcoming_processed)

# Add Home_win target variable
if 'Home_pts' in new_scores.columns and 'Visitor_pts' in new_scores.columns:
    new_scores['Home_win'] = (new_scores['Home_pts'] > new_scores['Visitor_pts']).astype(int)
    # Drop points columns to prevent data leakage in training
    new_scores = new_scores.drop(columns=['Home_pts', 'Visitor_pts'], errors='ignore')

# Remove points from upcoming games
upcoming_processed = upcoming_processed.drop(columns=['Home_pts', 'Visitor_pts'], errors='ignore')

# ---------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------

new_scores.to_excel('processed_data/2025_processed_scores.xlsx', index=False)
print('Saved processed_data/2025_processed_scores.xlsx')

upcoming_processed.to_excel('processed_data/2025_processed_upcoming_games.xlsx', index=False)
print('Saved processed_data/2025_processed_upcoming_games.xlsx')