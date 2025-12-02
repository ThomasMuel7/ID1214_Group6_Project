import pandas as pd
from tensorflow import keras
from test_data_prep import load_data

# Load the trained model
model = keras.models.load_model('./model/nfl_prediction_model4.keras')

# Load games to predict
# Change 'upcoming_games.csv' to your actual filename
games_to_predict = pd.read_csv('./data/2025_processed_upcoming_games.csv')
feature_cols = [col for col in games_to_predict.columns if col.endswith('_diff')]

# Extract features (same as training: all _diff columns)
X_predict = load_data(use_pca=True, n_components=30)

# Make predictions
predictions_prob = model.predict(X_predict, verbose=0)
predictions = (predictions_prob > 0.5).astype(int).flatten()

# Create results dataframe
results = games_to_predict[['Home', 'Visitor']].copy()
results['Model_Home_win'] = predictions
results['Model_home_win_probability'] = predictions_prob.flatten()

# Save predictions
results.to_excel('./data/predictions2.xlsx', index=False)
print(f"Predictions saved to ./data/predictions2.xlsx")
print(results.head())
