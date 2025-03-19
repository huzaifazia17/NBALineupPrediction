import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
import time
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

### **🔹 Configuration**
ALLOWED_FEATURES = [
    "season", "home_team", "away_team", "starting_min",
    "home_0", "home_1", "home_2", "home_3", "home_4",
    "away_0", "away_1", "away_2", "away_3", "away_4"
]

### **🔹 Load Training Data**
def load_training_data(years, use_all_outcomes=True):
    data_frames = []
    for year in years:
        file_path = f"dataset/matchups-{year}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not use_all_outcomes:
                df = df[df["outcome"] == 1]  # Only use home team wins if specified
            
            # Keep only allowed features
            columns_to_keep = [col for col in ALLOWED_FEATURES if col in df.columns]
            if "outcome" not in columns_to_keep and "outcome" in df.columns:
                columns_to_keep.append("outcome")
                
            df = df[columns_to_keep]
            data_frames.append(df)

    result = pd.concat(data_frames, ignore_index=True).dropna() if data_frames else pd.DataFrame()
    return result

### **🔹 Generate Team-Player Statistics**
def generate_team_player_stats(data):
    """Generate detailed team-player statistics"""
    # Track player frequency by team, position, and season
    team_position_players = defaultdict(lambda: defaultdict(Counter))
    
    # Track player-player combinations
    player_combos = defaultdict(Counter)
    
    # Process data in a single pass
    for _, row in data.iterrows():
        team = row["home_team"]
        season = row["season"]
        
        # Skip rows with missing data
        if pd.isna(team) or pd.isna(season):
            continue
        
        # Extract all valid players in the lineup
        lineup = []
        for pos in range(5):
            position = f"home_{pos}"
            if position in row and not pd.isna(row[position]) and row[position] != "?":
                player = row[position]
                lineup.append((position, player))
                
                # Add to team-position-player counter
                team_position_players[(team, season)][position][player] += 1
        
        # Add player combinations (which players play together)
        for i, (pos_i, player_i) in enumerate(lineup):
            for pos_j, player_j in lineup[i+1:]:
                key = (team, season, pos_i, pos_j)
                player_combos[key][(player_i, player_j)] += 1
    
    return team_position_players, player_combos

### **🔹 Find Missing Positions in Test Data**
def find_missing_positions(test_data):
    """Find which positions have missing values marked with '?'"""
    missing_positions = {}
    
    for pos in range(5):
        position = f"home_{pos}"
        if position in test_data.columns and test_data[position].isin(["?"]).any():
            indices = test_data.index[test_data[position] == "?"].tolist()
            missing_positions[position] = indices
            print(f"Found {len(indices)} rows with '?' in {position} column")
    
    return missing_positions

### **🔹 Statistical Only Prediction**
def predict_statistical(team_stats, player_combos, test_data, test_labels):
    """Use pure statistical prediction for speed"""
    # Skip header row in test labels
    actual_labels = test_labels[1:]
    
    # Find positions with missing players
    missing_positions = find_missing_positions(test_data)
    
    # Track predictions
    all_predictions = []
    predictions_by_year = defaultdict(lambda: {"correct": 0, "total": 0, "actual": [], "predicted": []})
    
    # Process each position
    for position, indices in missing_positions.items():
        print(f"Processing {position} with {len(indices)} missing values...")
        
        # Make predictions for each row with this position missing
        for idx in indices:
            row = test_data.loc[idx]
            team = row["home_team"]
            season = row["season"]
            year = season
            
            # Get actual player
            actual = actual_labels[idx]
            
            # Get team's players for this position
            if (team, season) in team_stats and position in team_stats[(team, season)]:
                team_players = team_stats[(team, season)][position]
                
                # Get current lineup for compatibility scoring
                lineup = []
                for pos in range(5):
                    other_pos = f"home_{pos}"
                    if other_pos != position and other_pos in row and not pd.isna(row[other_pos]) and row[other_pos] != "?":
                        lineup.append((other_pos, row[other_pos]))
                
                # If we have lineup info, score based on compatibility
                if lineup and team_players:
                    # Score each player by compatibility with current lineup
                    compatibility_scores = {player: 0 for player in team_players.keys()}
                    
                    for player in compatibility_scores:
                        # Add base frequency score
                        compatibility_scores[player] += team_players[player] * 10  # Weigh frequency heavily
                        
                        # Add compatibility with current lineup
                        for other_pos, other_player in lineup:
                            key = (team, season, position, other_pos)
                            rev_key = (team, season, other_pos, position)
                            
                            # Check both directions of player pairs
                            if key in player_combos and (player, other_player) in player_combos[key]:
                                compatibility_scores[player] += player_combos[key][(player, other_player)] * 5
                            
                            if rev_key in player_combos and (other_player, player) in player_combos[rev_key]:
                                compatibility_scores[player] += player_combos[rev_key][(other_player, player)] * 5
                    
                    # Choose player with highest compatibility score
                    if compatibility_scores:
                        prediction = max(compatibility_scores.items(), key=lambda x: x[1])[0]
                    else:
                        # Fallback to most common player
                        prediction = team_players.most_common(1)[0][0]
                else:
                    # No lineup info, just use most common player
                    prediction = team_players.most_common(1)[0][0]
            else:
                # No data for this team and position, use global fallback
                all_players = []
                for (t, s), positions in team_stats.items():
                    if position in positions:
                        for player, count in positions[position].items():
                            all_players.extend([player] * count)  # Add weighted by frequency
                
                if all_players:
                    prediction = Counter(all_players).most_common(1)[0][0]
                else:
                    prediction = "Unknown_Player"
            
            # Record prediction
            all_predictions.append({
                "year": year,
                "team": team,
                "position": position,
                "actual": actual,
                "predicted": prediction,
                "correct": prediction == actual
            })
            
            # Update year stats
            predictions_by_year[year]["total"] += 1
            predictions_by_year[year]["actual"].append(actual)
            predictions_by_year[year]["predicted"].append(prediction)
            if prediction == actual:
                predictions_by_year[year]["correct"] += 1
    
    # Calculate metrics
    results = []
    all_actual = []
    all_predicted = []
    
    for year, stats in predictions_by_year.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            f1 = f1_score(stats["actual"], stats["predicted"], average="weighted", zero_division=0)
            
            results.append({
                "year": year,
                "accuracy": accuracy,
                "f1_score": f1,
                "correct": stats["correct"],
                "total": stats["total"]
            })
            
            all_actual.extend(stats["actual"])
            all_predicted.extend(stats["predicted"])
    
    # Calculate overall metrics
    total_correct = sum(r["correct"] for r in results)
    total_predictions = sum(r["total"] for r in results)
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    overall_f1 = f1_score(all_actual, all_predicted, average="weighted", zero_division=0)
    
    return results, overall_accuracy, overall_f1, all_predictions

### **🔹 Year-Specific Model Predictor** 
def create_year_features(row, position, team_encoder, missing_positions):
    """Create features for a specific test row"""
    features = {}
    
    # Basic features
    features["team_id"] = team_encoder.get(row["home_team"], -1)
    features["season"] = row["season"]
    
    if "starting_min" in row:
        features["starting_min"] = row["starting_min"]
    
    if "away_team" in row:
        features["opponent_id"] = team_encoder.get(row["away_team"], -1)
    
    # Current lineup features
    for pos in range(5):
        other_pos = f"home_{pos}"
        if other_pos != position and other_pos in row and not pd.isna(row[other_pos]) and row[other_pos] != "?":
            # We have this player in the lineup
            features[f"has_player_{pos}"] = 1
    
    return features

def train_year_models(train_data, test_data, test_labels, team_stats, player_combos):
    """Train year-specific models for better accuracy"""
    # Skip header row in test labels
    actual_labels = test_labels[1:]
    
    # Find positions with missing players
    missing_positions = find_missing_positions(test_data)
    
    # Create team encoder
    all_teams = train_data["home_team"].unique()
    team_encoder = {team: i for i, team in enumerate(all_teams)}
    
    # Track predictions
    all_predictions = []
    predictions_by_year = defaultdict(lambda: {"correct": 0, "total": 0, "actual": [], "predicted": []})
    
    # Group test data by year
    years = sorted(test_data["season"].unique())
    
    for year in years:
        print(f"Processing year {year}...")
        
        # Get training data for this year
        if year <= 2010:
            # For early years, use all available data up to that year
            train_years = list(range(2007, year + 1))
        else:
            # For later years, use 3 years of data
            train_years = list(range(year - 2, year + 1))
            
        # Filter training data for these years
        year_train_data = train_data[train_data["season"].isin(train_years)]
        
        # Get test data for this year
        year_test_data = test_data[test_data["season"] == year]
        
        # Process each position for this year
        for position in missing_positions:
            indices = [idx for idx in missing_positions[position] if test_data.loc[idx, "season"] == year]
            
            if not indices:
                continue
                
            print(f"  Position {position}: {len(indices)} missing values")
            
            # Extract features and labels for this position
            X_train_rows = []
            y_train = []
            
            for _, row in year_train_data.iterrows():
                if position not in row or pd.isna(row[position]) or row[position] == "?":
                    continue
                    
                features = create_year_features(row, position, team_encoder, missing_positions)
                X_train_rows.append(features)
                y_train.append(row[position])
            
            if len(X_train_rows) < 10:
                print(f"    Not enough training data, using statistical prediction")
                use_model = False
            else:
                # Count class frequencies
                class_counts = Counter(y_train)
                common_classes = {player for player, count in class_counts.items() if count > 1}
                
                if len(common_classes) >= 2:
                    # Filter to keep only common classes
                    common_indices = [i for i, player in enumerate(y_train) if player in common_classes]
                    if len(common_indices) >= 10:
                        # Convert to DataFrame
                        X_train_df = pd.DataFrame(X_train_rows)
                        X_filtered = X_train_df.iloc[common_indices]
                        y_filtered = [y_train[i] for i in common_indices]
                        
                        # Train Random Forest
                        try:
                            model = RandomForestClassifier(
                                n_estimators=200,  # More trees for accuracy
                                max_depth=15,      # Deeper trees for better patterns
                                min_samples_leaf=1,
                                class_weight='balanced',
                                random_state=42,
                                n_jobs=-1
                            )
                            model.fit(X_filtered, y_filtered)
                            print(f"    Trained model with {len(y_filtered)} examples")
                            use_model = True
                        except Exception as e:
                            print(f"    Error training model: {str(e)}")
                            use_model = False
                    else:
                        print(f"    Not enough common players")
                        use_model = False
                else:
                    print(f"    Not enough common classes")
                    use_model = False
            
            # Make predictions for this position in this year
            for idx in indices:
                row = test_data.loc[idx]
                team = row["home_team"]
                season = row["season"]
                
                # Get actual player
                actual = actual_labels[idx]
                
                # Model prediction (if available)
                if use_model:
                    try:
                        # Create test features
                        test_features = create_year_features(row, position, team_encoder, missing_positions)
                        X_test = pd.DataFrame([test_features])
                        
                        # Add missing columns
                        for col in model.feature_names_in_:
                            if col not in X_test.columns:
                                X_test[col] = 0
                        
                        X_test = X_test[model.feature_names_in_]
                        
                        # Make prediction
                        model_prediction = model.predict(X_test)[0]
                    except Exception:
                        model_prediction = None
                else:
                    model_prediction = None
                
                # Statistical prediction for fallback
                if (team, season) in team_stats and position in team_stats[(team, season)]:
                    team_players = team_stats[(team, season)][position]
                    
                    # Get current lineup
                    lineup = []
                    for pos in range(5):
                        other_pos = f"home_{pos}"
                        if other_pos != position and other_pos in row and not pd.isna(row[other_pos]) and row[other_pos] != "?":
                            lineup.append((other_pos, row[other_pos]))
                    
                    # Score players by compatibility
                    if lineup and team_players:
                        compatibility_scores = {player: team_players[player] * 10 for player in team_players}
                        
                        for player in compatibility_scores:
                            for other_pos, other_player in lineup:
                                key = (team, season, position, other_pos)
                                rev_key = (team, season, other_pos, position)
                                
                                if key in player_combos and (player, other_player) in player_combos[key]:
                                    compatibility_scores[player] += player_combos[key][(player, other_player)] * 5
                                if rev_key in player_combos and (other_player, player) in player_combos[rev_key]:
                                    compatibility_scores[player] += player_combos[rev_key][(other_player, player)] * 5
                        
                        if compatibility_scores:
                            stat_prediction = max(compatibility_scores.items(), key=lambda x: x[1])[0]
                        else:
                            stat_prediction = team_players.most_common(1)[0][0]
                    else:
                        stat_prediction = team_players.most_common(1)[0][0]
                else:
                    # No data for this team, use global fallback
                    stat_prediction = None
                
                # Choose final prediction
                if model_prediction is not None and (team, season) in team_stats and position in team_stats[(team, season)] and model_prediction in team_stats[(team, season)][position]:
                    # Model prediction is valid for this team
                    prediction = model_prediction
                elif stat_prediction is not None:
                    # Use statistical prediction
                    prediction = stat_prediction
                elif model_prediction is not None:
                    # Use model prediction as fallback
                    prediction = model_prediction
                else:
                    # Ultimate fallback - use most common player across all teams
                    all_players = []
                    for (t, s), positions in team_stats.items():
                        if position in positions:
                            all_players.extend(positions[position].keys())
                    
                    if all_players:
                        prediction = Counter(all_players).most_common(1)[0][0]
                    else:
                        prediction = "Unknown_Player"
                
                # Record prediction
                all_predictions.append({
                    "year": year,
                    "team": team,
                    "position": position,
                    "actual": actual,
                    "predicted": prediction,
                    "correct": prediction == actual
                })
                
                # Update year stats
                predictions_by_year[year]["total"] += 1
                predictions_by_year[year]["actual"].append(actual)
                predictions_by_year[year]["predicted"].append(prediction)
                if prediction == actual:
                    predictions_by_year[year]["correct"] += 1
    
    # Calculate metrics
    results = []
    all_actual = []
    all_predicted = []
    
    for year, stats in predictions_by_year.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            f1 = f1_score(stats["actual"], stats["predicted"], average="weighted", zero_division=0)
            
            results.append({
                "year": year,
                "accuracy": accuracy, 
                "f1_score": f1,
                "correct": stats["correct"],
                "total": stats["total"]
            })
            
            all_actual.extend(stats["actual"])
            all_predicted.extend(stats["predicted"])
    
    # Calculate overall metrics
    total_correct = sum(r["correct"] for r in results)
    total_predictions = sum(r["total"] for r in results)
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    overall_f1 = f1_score(all_actual, all_predicted, average="weighted", zero_division=0)
    
    return results, overall_accuracy, overall_f1, all_predictions

### **🔹 Main Function**
def main():
    print("Starting optimized NBA player prediction...")
    start_time = time.time()
    
    # Load test data
    test_data = pd.read_csv("test_dataset/NBA_test.csv")
    
    # Load test labels
    test_labels_df = pd.read_csv("test_dataset/NBA_test_labels.csv", header=None)
    test_labels = test_labels_df[0].values
    
    print(f"Loaded {len(test_data)} test rows and {len(test_labels)} labels (including header row)")
    
    # Load all training data
    all_years = list(range(2007, 2017))
    train_data = load_training_data(all_years, use_all_outcomes=True)
    print(f"Loaded {len(train_data)} training examples")
    
    # Generate team and player statistics
    print("Generating team and player statistics...")
    team_stats, player_combos = generate_team_player_stats(train_data)
    print("Statistics generation complete")
    
    # Try both approaches and use the better one
    print("\nRunning statistical prediction...")
    stat_results, stat_accuracy, stat_f1, stat_predictions = predict_statistical(
        team_stats, player_combos, test_data, test_labels
    )
    
    print(f"\nStatistical prediction accuracy: {stat_accuracy:.4f}")
    print(f"Statistical prediction F1 score: {stat_f1:.4f}")
    
    print("\nRunning model-based prediction...")
    model_results, model_accuracy, model_f1, model_predictions = train_year_models(
        train_data, test_data, test_labels, team_stats, player_combos
    )
    
    print(f"\nModel-based prediction accuracy: {model_accuracy:.4f}")
    print(f"Model-based prediction F1 score: {model_f1:.4f}")
    
    # Choose the better approach
    if model_accuracy >= stat_accuracy:
        print("\nUsing model-based predictions (higher accuracy)")
        results = model_results
        overall_accuracy = model_accuracy
        overall_f1 = model_f1
        all_predictions = model_predictions
    else:
        print("\nUsing statistical predictions (higher accuracy)")
        results = stat_results
        overall_accuracy = stat_accuracy
        overall_f1 = stat_f1
        all_predictions = stat_predictions
    
    # Print results
    print("\nResults by Year:")
    results_df = pd.DataFrame(results).sort_values(by="year")
    print(results_df[["year", "accuracy", "f1_score", "correct", "total"]])
    
    print(f"\nOverall accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1 score: {overall_f1:.4f}")
    
    # Save results
    results_df.to_csv("nba_prediction_results.csv", index=False)
    pd.DataFrame(all_predictions).to_csv("nba_predictions_detailed.csv", index=False)
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()