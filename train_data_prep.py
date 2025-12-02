import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(use_pca=False, n_components=30):   
    # Load processed scores with team statistics
    scores = pd.read_csv('data/2025_processed_scores.csv')

    # Prepare data for neural network
    # Drop rows with missing values
    scores_clean = scores.dropna()

    # Extract features (all _diff columns) and target (Home_win)
    feature_cols = [col for col in scores_clean.columns if col.endswith('_diff')]

    X = scores_clean[feature_cols].values
    y = scores_clean['Home_win'].values
        
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA if requested
    if use_pca:
        pca = PCA(n_components=n_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, len(feature_cols) if not use_pca else n_components