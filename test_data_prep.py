import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(use_pca=False, n_components=30):   
    # Load processed scores with team statistics
    scores = pd.read_csv('data/2025_processed_upcoming_games.csv')

    # Prepare data for neural network
    # Drop rows with missing values
    scores_clean = scores.dropna()

    # Extract features (all _diff columns) and target (Home_win)
    feature_cols = [col for col in scores_clean.columns if col.endswith('_diff')]

    X = scores_clean[feature_cols].values
        
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if use_pca:
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
    return X_scaled