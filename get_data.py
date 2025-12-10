import requests
import pandas as pd
from bs4 import BeautifulSoup
import io
from datetime import date, timedelta
import argparse
import os
import time

os.makedirs("prep_data", exist_ok=True)

# --- 1. CONFIGURATION & HEADERS ---

def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

# --- 2. CLEANING HELPERS ---

VALID_TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
    "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
]

def clean_team_name_str(name):
    """Cleans a team name string. Example: 'Arizona CardinalsARI' -> 'Arizona Cardinals'"""
    name_str = str(name)
    for valid_team in VALID_TEAMS:
        if valid_team in name_str:
            return valid_team
    return name_str.strip()

def get_clean_team_name_html(td_element):
    """Extracts team name from HTML, preferring the desktop view span."""
    full_name_span = td_element.find('span', class_='d-none d-xl-inline')
    if full_name_span:
        return full_name_span.get_text(strip=True)
    return td_element.get_text(strip=True)

def get_week_number(header_text):
    try:
        return int(header_text.replace("Week", "").strip())
    except ValueError:
        return 0

def convert_fraction_to_float(val):
    """Converts strings like '20/21' to float (0.952). Returns original val if not a fraction."""
    if isinstance(val, str) and '/' in val:
        try:
            parts = val.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                return num / den if den != 0 else 0.0
        except ValueError:
            pass
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

# --- 3. SCRAPING FUNCTIONS ---

def scrape_nfl_type_category_team_stats(type, category):
    url = f"https://www.footballdb.com/statistics/nfl/team-stats/{type}-{category}/2025/regular-season"
    output_filename = f'./prep_data/2025_{category}_{type}.xlsx'
    
    try:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        target_table = soup.select_one('table.statistics.scrollable')
        
        if target_table:
            df_list = pd.read_html(io.StringIO(str(target_table)))
            if df_list:
                df = df_list[0]
                
                # Flatten MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [' '.join([str(c) for c in col if "Unnamed" not in str(c)]).strip() for col in df.columns.values]
                
                # 1. Clean Team Names (First Column)
                if not df.empty:
                    team_col_name = df.columns[0]
                    df[team_col_name] = df[team_col_name].apply(clean_team_name_str)
                
                # 2. Specific Data Type Cleaning
                
                # Rule: For 'scoring', convert 'FG', 'PAT', 'XP' from '20/21' to float
                if category == 'scoring':
                    # Identify likely columns based on name containing FG, PAT, or XP
                    target_cols = [c for c in df.columns if any(sub in c.upper() for sub in ['FG', 'PAT', 'XP', 'XPT'])]
                    for col in target_cols:
                        df[col] = df[col].apply(convert_fraction_to_float)
                
                # Rule: For returns, remove 't' from 'Lg' columns and make int
                if category in ['kickoff-returns', 'punt-returns']:
                    # Identify likely columns ending in 'Lg' or named 'Lg'
                    lg_cols = [c for c in df.columns if c == 'Lg' or c.endswith(' Lg') or c.endswith('Lg')]
                    for col in lg_cols:
                        # Remove 't' (case insensitive)
                        df[col] = df[col].astype(str).str.replace('t', '', case=False, regex=False)
                        # Convert to int (via float to handle NaNs/strings safely)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

                df.to_excel(output_filename, index=False)
                print(f"Saved: {output_filename}")
            else:
                print(f"Error ({type}-{category}): Pandas could not interpret table data.")
        else:
            print(f"Error ({type}-{category}): Statistics table not found.")
            
    except Exception as e:
        print(f"Error ({type}-{category}): {e}")
    
    time.sleep(1)

def parse_html_data(url="https://www.footballdb.com/games/index.html"):
    weeks_data = {}
    try:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        week_divs = soup.find_all('div', class_='ltbluediv')
        
        for div in week_divs:
            header = div.find('span', class_='divheader')
            if not header:
                continue   
            week_num = get_week_number(header.get_text())
            table = div.find_next_sibling('table', class_='statistics') 
            
            if table:
                games = []
                rows = table.find('tbody').find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if not cols:
                        continue
                    
                    date_span = cols[0].find('span', class_='d-none d-xl-inline')
                    date_text = date_span.get_text(strip=True) if date_span else cols[0].get_text(strip=True) 
                    
                    game = {
                        'Week': week_num,
                        'Date': date_text,
                        'Visitor': get_clean_team_name_html(cols[1]),
                        'Visitor Score': cols[2].get_text(strip=True),
                        'Home': get_clean_team_name_html(cols[3]),
                        'Home Score': cols[4].get_text(strip=True)
                    }
                    games.append(game)
                weeks_data[week_num] = games
    except Exception as e:
        print(f"Error parsing HTML data: {e}") 
    return weeks_data

# --- 4. SAVING FUNCTIONS ---

def save_training_scores(weeks_data, final_week=14, filename="prep_data/2025_scores.xlsx"):
    all_games = []
    for week, games in weeks_data.items():
        if 1 <= week <= final_week:
            all_games.extend(games)    
    df = pd.DataFrame(all_games)
    if not df.empty:
        df = df[['Week', 'Date', 'Visitor', 'Visitor Score', 'Home', 'Home Score']]
        df = df.rename(columns={'Visitor Score': 'Visitor_pts', 'Home Score': 'Home_pts'})
        
        df['Visitor'] = df['Visitor'].apply(clean_team_name_str)
        df['Home'] = df['Home'].apply(clean_team_name_str)
        
        df.to_excel(filename, index=False)
        print(f"Saved: {filename}")
    else:
        print(f"Warning: No data found for training scores (Weeks 1-{final_week})")

def save_upcoming_teams(weeks_data, first_week=15, filename="prep_data/2025_upcoming_games.xlsx"):
    all_games = []
    max_week = max(weeks_data.keys()) if weeks_data else 0
    for week in range(first_week, max_week + 1):
        if week in weeks_data:
            all_games.extend(weeks_data[week])         
    df = pd.DataFrame(all_games)
    if not df.empty:
        df = df[['Week', 'Date', 'Visitor', 'Home']]
        
        df['Visitor'] = df['Visitor'].apply(clean_team_name_str)
        df['Home'] = df['Home'].apply(clean_team_name_str)
        
        df.to_excel(filename, index=False)
        print(f"Saved: {filename}")
    else:
        print(f"No upcoming games found starting from week {first_week}.")

def save_upcoming_scores(weeks_data, first_week=15, filename="prep_data/2025_upcoming_scores.xlsx"):
    all_games = []
    max_week = max(weeks_data.keys()) if weeks_data else 0
    for week in range(first_week, max_week + 1):
        if week in weeks_data:
            all_games.extend(weeks_data[week])    
    df = pd.DataFrame(all_games)
    
    if not df.empty:
        yesterday = pd.Timestamp(date.today() - timedelta(days=1)).normalize()
        df['Date_Obj'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce') 
        
        df_target = df[df['Date_Obj'] <= yesterday].copy()
        
        if not df_target.empty:
            df_target = df_target[['Week', 'Date', 'Visitor', 'Visitor Score', 'Home', 'Home Score']]
            df_target = df_target.rename(columns={'Visitor Score': 'Visitor_pts', 'Home Score': 'Home_pts'})
            
            df_target['Visitor'] = df_target['Visitor'].apply(clean_team_name_str)
            df_target['Home'] = df_target['Home'].apply(clean_team_name_str)
            
            df_target.to_excel(filename, index=False)
            print(f"Saved: {filename}")
        else:
            print(f"No played games found from week {first_week} until yesterday ({yesterday.date()}).")
    else:
        print(f"No game data available starting from week {first_week}.")

# --- 5. MAIN ---

def scrape_nfl_scores(mode='all', week=14):
    print("Scraping NFL Scores and Schedule...")
    data = parse_html_data()
    if not data:
        print("No data retrieved from schedule page.")
        return
    
    if mode in ['all', 'training']:   
        save_training_scores(data, week)
    
    if mode in ['all', 'upcoming']:
        save_upcoming_teams(data, week+1)
        save_upcoming_scores(data, week+1)
        
def scrape_nfl_team_stats():
    print("Scraping NFL Team Stats...")
    categories = ["totals", "passing", "rushing", "kickoff-returns", "punt-returns", "punting", "scoring", "downs"]
    types = ["offense", "defense"]
    for category in categories:
        for type in types:
            scrape_nfl_type_category_team_stats(type, category)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NFL 2025 Data")
    parser.add_argument('--week', type=int, default=14, help='The last completed week for training data (default: 14)')
    parser.add_argument('--mode', type=str, choices=['all', 'stats', 'scores'], default='all', help='What to scrape')
    parser.add_argument('--type', type=str, choices=['all', 'training', 'upcoming'], default='all', help='For scores mode: training, upcoming, or all')
    
    args = parser.parse_args()
    
    if args.mode in ['all', 'stats']:
        scrape_nfl_team_stats()
        
    if args.mode in ['all', 'scores']:
        scrape_nfl_scores(mode=args.type, week=args.week)