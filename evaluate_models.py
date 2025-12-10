import pandas as pd

upcoming_games = pd.read_excel('prep_data/2025_upcoming_scores.xlsx')
hybrid = pd.read_excel('predictions/hybrid_predictions.xlsx')
classical = pd.read_excel('predictions/classical_predictions.xlsx')
quantum = pd.read_excel('predictions/quantum_predictions.xlsx')
classical_pca = pd.read_excel('predictions/pca_classical_predictions.xlsx')
hybrid_pca = pd.read_excel('predictions/hybrid_pca_predictions.xlsx')


upcoming_games['Actual_Home_win'] = (upcoming_games['Home_pts'] > upcoming_games['Visitor_pts']).astype(int)

models = {
    "Classical": classical,
    "Hybrid": hybrid,
    "Quantum": quantum,
    "Classical PCA": classical_pca,
    "Hybrid PCA": hybrid_pca
}

for name, df_pred in models.items():
    merged = pd.merge(upcoming_games, df_pred, on=['Home', 'Visitor'], how='inner')
    correct_predictions = merged[merged['Actual_Home_win'] == merged['Model_Home_win']]
    acc = len(correct_predictions) / len(merged)
    print(f"{name} Model Accuracy: {acc:.2%} ({len(correct_predictions)}/{len(merged)})")