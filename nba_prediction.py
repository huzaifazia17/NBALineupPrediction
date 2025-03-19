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

### **🔹 Generate Team-Player Statistics**
def generate_team_player_stats(data):
    """Generate detailed team-player statistics for better prediction"""
    # Track player frequency by team, position, and season
    team_position_players = defaultdict(lambda: defaultdict(Counter))
    
    # Track player pairs (which players play together)
    player_pairs = defaultdict(Counter)
    
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
        
        # Add player pairs to counter (how often players appear together)
        for i, (pos_i, player_i) in enumerate(lineup):
            for pos_j, player_j in lineup[i+1:]:
                key = (team, season, pos_i, pos_j)
                player_pairs[key][(player_i, player_j)] += 1
    
    return team_position_players, player_pairs

### **🔹 Specialized 2016 Strategy**
def create_2016_features(row, position, training_data):
    """Create specialized features for 2016 predictions"""
    features = {}
    
    # Base features
    features["team_id"] = hash(row["home_team"]) % 10000  # Finer-grained hash
    features["season"] = row["season"]
    
    if "starting_min" in row:
        features["starting_min"] = row["starting_min"]
    
    if "away_team" in row:
        features["opponent_id"] = hash(row["away_team"]) % 10000
    
    # Add lineup features (which positions are filled)
    for pos in range(5):
        other_pos = f"home_{pos}"
        if other_pos != position and other_pos in row and not pd.isna(row[other_pos]) and row[other_pos] != "?":
            features[f"has_player_{pos}"] = 1
            
            # Add actual player information for special 2016 handling
            player = row[other_pos]
            player_id = hash(player) % 10000
            features[f"player_{pos}_id"] = player_id
    
    # Use 2015 data to help with 2016 predictions
    if 2015 in training_data:
        df_2015 = training_data[2015]
        team_matches = df_2015[df_2015["home_team"] == row["home_team"]]
        
        if len(team_matches) > 0:
            # Count players in this position for this team in 2015
            if position in team_matches.columns:
                player_counts = team_matches[position].value_counts()
                features["team_2015_players"] = len(player_counts)
    
    return features

### **🔹 Hybrid Prediction Strategy**
def predict_hybrid(train_data, test_data, test_labels):
    """Hybrid approach optimized for all years including 2016"""
    # Skip header row in test labels
    actual_labels = test_labels[1:]
    
    # Find positions with missing players
    missing_positions = find_missing_positions(test_data)
    
    # Generate team-player statistics
    print("Generating team and player statistics...")
    team_stats, player_pairs = generate_team_player_stats(train_data)
    
    # Group training data by year for specialized handling
    training_by_year = {}
    for year in range(2007, 2017):
        training_by_year[year] = train_data[train_data["season"] == year]
    
    # Track predictions
    all_predictions = []
    predictions_by_year = defaultdict(lambda: {"correct": 0, "total": 0, "actual": [], "predicted": []})
    
    # Process each position
    for position, indices in missing_positions.items():
        print(f"Processing {position} with {len(indices)} missing values...")
        
        # Group indices by year for specialized handling
        indices_by_year = defaultdict(list)
        for idx in indices:
            year = test_data.loc[idx, "season"]
            indices_by_year[year].append(idx)
        
        # Process 2007-2015 using team-specific approach
        for year in range(2007, 2016):
            if year not in indices_by_year or not indices_by_year[year]:
                continue
                
            print(f"  Processing year {year} with {len(indices_by_year[year])} examples")
            
            # Select training years for this prediction year
            if year <= 2010:
                train_years = list(range(2007, year + 1))
            else:
                train_years = list(range(year - 2, year + 1))
            
            # Prepare training data for this year
            year_training = train_data[train_data["season"].isin(train_years)]
            
            # Train team-specific models for this year
            team_models = {}
            team_training_data = defaultdict(list)
            team_training_labels = defaultdict(list)
            
            for _, row in year_training.iterrows():
                if position not in row or pd.isna(row[position]) or row[position] == "?":
                    continue
                
                team = row["home_team"]
                # Create basic features
                features = {
                    "team_id": hash(team) % 1000,
                    "season": row["season"]
                }
                
                if "starting_min" in row:
                    features["starting_min"] = row["starting_min"]
                
                if "away_team" in row:
                    features["opponent_id"] = hash(row["away_team"]) % 1000
                
                # Add lineup features
                for pos in range(5):
                    other_pos = f"home_{pos}"
                    if other_pos != position and other_pos in row and not pd.isna(row[other_pos]) and row[other_pos] != "?":
                        features[f"has_player_{pos}"] = 1
                
                team_training_data[team].append(features)
                team_training_labels[team].append(row[position])
            
            # Train models for teams with enough data
            for team, features in team_training_data.items():
                if len(features) >= 10:
                    labels = team_training_labels[team]
                    
                    # Count player frequencies
                    player_counts = Counter(labels)
                    common_players = {player for player, count in player_counts.items() if count > 1}
                    
                    if len(common_players) >= 2:
                        # Keep only common players
                        common_indices = [i for i, player in enumerate(labels) if player in common_players]
                        if len(common_indices) >= 10:
                            X = pd.DataFrame(features).iloc[common_indices]
                            y = [labels[i] for i in common_indices]
                            
                            try:
                                model = RandomForestClassifier(
                                    n_estimators=150,
                                    max_depth=15,
                                    min_samples_leaf=2,
                                    class_weight='balanced',
                                    random_state=42
                                )
                                model.fit(X, y)
                                team_models[team] = (model, X.columns.tolist())
                            except Exception as e:
                                print(f"    Error training model for {team}: {str(e)}")
            
            # Train global model for this year and position
            global_features = []
            global_labels = []
            
            for team, features_list in team_training_data.items():
                global_features.extend(features_list)
                global_labels.extend(team_training_labels[team])
            
            global_model = None
            global_columns = None
            
            if len(global_features) >= 20:
                player_counts = Counter(global_labels)
                common_players = {player for player, count in player_counts.items() if count > 1}
                
                if len(common_players) >= 2:
                    common_indices = [i for i, player in enumerate(global_labels) if player in common_players]
                    if len(common_indices) >= 20:
                        X_global = pd.DataFrame(global_features).iloc[common_indices]
                        y_global = [global_labels[i] for i in common_indices]
                        
                        try:
                            global_model = RandomForestClassifier(
                                n_estimators=200,
                                max_depth=15,
                                min_samples_leaf=1,
                                class_weight='balanced',
                                random_state=42
                            )
                            global_model.fit(X_global, y_global)
                            global_columns = X_global.columns.tolist()
                            print(f"    Trained global model with {len(y_global)} examples")
                        except Exception as e:
                            print(f"    Error training global model: {str(e)}")
            
            # Make predictions for this year
            for idx in indices_by_year[year]:
                row = test_data.loc[idx]
                team = row["home_team"]
                actual = actual_labels[idx]
                
                # Create test features
                features = {
                    "team_id": hash(team) % 1000,
                    "season": row["season"]
                }
                
                if "starting_min" in row:
                    features["starting_min"] = row["starting_min"]
                
                if "away_team" in row:
                    features["opponent_id"] = hash(row["away_team"]) % 1000
                
                for pos in range(5):
                    other_pos = f"home_{pos}"
                    if other_pos != position and other_pos in row and not pd.isna(row[other_pos]) and row[other_pos] != "?":
                        features[f"has_player_{pos}"] = 1
                
                # Try team model first
                team_prediction = None
                if team in team_models:
                    model, columns = team_models[team]
                    try:
                        # Ensure test features match training features
                        X_test = pd.DataFrame([features])
                        for col in columns:
                            if col not in X_test.columns:
                                X_test[col] = 0
                        X_test = X_test[columns]
                        
                        team_prediction = model.predict(X_test)[0]
                    except Exception:
                        team_prediction = None
                
                # Try global model if team model failed
                global_prediction = None
                if global_model is not None and global_columns is not None:
                    try:
                        # Ensure test features match training features
                        X_test = pd.DataFrame([features])
                        for col in global_columns:
                            if col not in X_test.columns:
                                X_test[col] = 0
                        X_test = X_test[global_columns]
                        
                        global_prediction = global_model.predict(X_test)[0]
                    except Exception:
                        global_prediction = None
                
                # Statistical fallback
                stat_prediction = None
                if (team, year) in team_stats and position in team_stats[(team, year)]:
                    player_counts = team_stats[(team, year)][position]
                    if player_counts:
                        stat_prediction = player_counts.most_common(1)[0][0]
                
                # Choose final prediction (prioritize team model)
                if team_prediction is not None:
                    prediction = team_prediction
                elif global_prediction is not None:
                    prediction = global_prediction
                elif stat_prediction is not None:
                    prediction = stat_prediction
                else:
                    # Ultimate fallback - most common player for this position
                    all_players = []
                    for (t, s), positions in team_stats.items():
                        if position in positions:
                            all_players.extend(positions[position].keys())
                    
                    prediction = Counter(all_players).most_common(1)[0][0] if all_players else "Unknown_Player"
                
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
        
        # Special handling for 2016
        if 2016 in indices_by_year and indices_by_year[2016]:
            print(f"  Special processing for 2016 with {len(indices_by_year[2016])} examples")
            
            # Use 2015 data to help with 2016
            train_years = [2014, 2015]
            year_training = train_data[train_data["season"].isin(train_years)]
            
            # Create specialized features focused on 2016
            features_2016 = []
            labels_2016 = []
            
            for _, row in year_training.iterrows():
                if position not in row or pd.isna(row[position]) or row[position] == "?":
                    continue
                
                features = create_2016_features(row, position, training_by_year)
                features_2016.append(features)
                labels_2016.append(row[position])
            
            # Train a specialized model for 2016
            if len(features_2016) >= 20:
                # Create DataFrame
                X_2016 = pd.DataFrame(features_2016)
                y_2016 = labels_2016
                
                # Train model
                try:
                    model_2016 = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=20,
                        min_samples_leaf=1,
                        class_weight='balanced',
                        random_state=42
                    )
                    model_2016.fit(X_2016, y_2016)
                    model_columns = X_2016.columns.tolist()
                    print(f"    Trained 2016 model with {len(y_2016)} examples")
                    
                    # Make predictions for 2016
                    for idx in indices_by_year[2016]:
                        row = test_data.loc[idx]
                        team = row["home_team"]
                        actual = actual_labels[idx]
                        
                        # Create specialized 2016 features
                        test_features = create_2016_features(row, position, training_by_year)
                        
                        # Prepare test features
                        X_test = pd.DataFrame([test_features])
                        for col in model_columns:
                            if col not in X_test.columns:
                                X_test[col] = 0
                        X_test = X_test[model_columns]
                        
                        # Make prediction
                        try:
                            model_prediction = model_2016.predict(X_test)[0]
                            
                            # Statistical fallback for validation
                            stat_prediction = None
                            if (team, 2015) in team_stats and position in team_stats[(team, 2015)]:
                                # Try 2015 data for this team
                                player_counts = team_stats[(team, 2015)][position]
                                if player_counts:
                                    stat_prediction = player_counts.most_common(1)[0][0]
                            
                            # Use model prediction if it exists in team history
                            if stat_prediction is not None and model_prediction == stat_prediction:
                                # Model agrees with statistical prediction - good sign
                                prediction = model_prediction
                            elif (team, 2015) in team_stats and position in team_stats[(team, 2015)] and model_prediction in team_stats[(team, 2015)][position]:
                                # Model prediction exists in team history
                                prediction = model_prediction
                            elif stat_prediction is not None:
                                # Use statistical prediction as fallback
                                prediction = stat_prediction
                            else:
                                # Default to model prediction
                                prediction = model_prediction
                            
                            # Record prediction
                            all_predictions.append({
                                "year": 2016,
                                "team": team,
                                "position": position,
                                "actual": actual,
                                "predicted": prediction,
                                "correct": prediction == actual
                            })
                            
                            # Update year stats
                            predictions_by_year[2016]["total"] += 1
                            predictions_by_year[2016]["actual"].append(actual)
                            predictions_by_year[2016]["predicted"].append(prediction)
                            if prediction == actual:
                                predictions_by_year[2016]["correct"] += 1
                                
                        except Exception as e:
                            print(f"    Error predicting: {str(e)}")
                            
                            # Fallback to statistical prediction
                            if (team, 2015) in team_stats and position in team_stats[(team, 2015)]:
                                # Try 2015 data for this team
                                player_counts = team_stats[(team, 2015)][position]
                                if player_counts:
                                    prediction = player_counts.most_common(1)[0][0]
                                else:
                                    prediction = "Unknown_Player"
                            else:
                                prediction = "Unknown_Player"
                            
                            # Record prediction
                            all_predictions.append({
                                "year": 2016,
                                "team": team,
                                "position": position,
                                "actual": actual,
                                "predicted": prediction,
                                "correct": prediction == actual
                            })
                            
                            # Update year stats
                            predictions_by_year[2016]["total"] += 1
                            predictions_by_year[2016]["actual"].append(actual)
                            predictions_by_year[2016]["predicted"].append(prediction)
                            if prediction == actual:
                                predictions_by_year[2016]["correct"] += 1
                            
                except Exception as e:
                    print(f"    Error training 2016 model: {str(e)}")
                    
                    # Handle all 2016 cases with statistical prediction
                    for idx in indices_by_year[2016]:
                        row = test_data.loc[idx]
                        team = row["home_team"]
                        actual = actual_labels[idx]
                        
                        # Use 2015 data for prediction
                        if (team, 2015) in team_stats and position in team_stats[(team, 2015)]:
                            player_counts = team_stats[(team, 2015)][position]
                            if player_counts:
                                prediction = player_counts.most_common(1)[0][0]
                            else:
                                prediction = "Unknown_Player"
                        else:
                            prediction = "Unknown_Player"
                        
                        # Record prediction
                        all_predictions.append({
                            "year": 2016,
                            "team": team,
                            "position": position,
                            "actual": actual,
                            "predicted": prediction,
                            "correct": prediction == actual
                        })
                        
                        # Update year stats
                        predictions_by_year[2016]["total"] += 1
                        predictions_by_year[2016]["actual"].append(actual)
                        predictions_by_year[2016]["predicted"].append(prediction)
                        if prediction == actual:
                            predictions_by_year[2016]["correct"] += 1
            else:
                # Not enough training data for 2016 model
                print("    Not enough training data for 2016 model, using statistical prediction")
                
                # Use statistical prediction for all 2016 cases
                for idx in indices_by_year[2016]:
                    row = test_data.loc[idx]
                    team = row["home_team"]
                    actual = actual_labels[idx]
                    
                    # Use 2015 data for prediction
                    if (team, 2015) in team_stats and position in team_stats[(team, 2015)]:
                        player_counts = team_stats[(team, 2015)][position]
                        if player_counts:
                            prediction = player_counts.most_common(1)[0][0]
                        else:
                            prediction = "Unknown_Player"
                    else:
                        prediction = "Unknown_Player"
                    
                    # Record prediction
                    all_predictions.append({
                        "year": 2016,
                        "team": team,
                        "position": position,
                        "actual": actual,
                        "predicted": prediction,
                        "correct": prediction == actual
                    })
                    
                    # Update year stats
                    predictions_by_year[2016]["total"] += 1
                    predictions_by_year[2016]["actual"].append(actual)
                    predictions_by_year[2016]["predicted"].append(prediction)
                    if prediction == actual:
                        predictions_by_year[2016]["correct"] += 1
    
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
    print("Starting hybrid NBA player prediction...")
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
    
    # Run hybrid prediction
    results, overall_accuracy, overall_f1, all_predictions = predict_hybrid(
        train_data, test_data, test_labels
    )
    
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