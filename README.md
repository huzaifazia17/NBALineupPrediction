# NBA Lineup Player Prediction

## Project Objectives

This project aims to predict missing players in NBA lineups by analyzing historical matchup data. The specific objectives include:

1. Predicting which player would fill a missing position (marked by "?") in an NBA lineup
2. Developing models that can accurately predict players across different seasons (2007-2016)
3. Creating a robust prediction system that handles the challenges of player movement between teams
4. Achieving high accuracy in player predictions despite the complexity of NBA roster changes

## Model Approach

The project implements a hybrid prediction approach combining multiple techniques:

### Data Processing
- Uses historical matchup data from 2007-2016
- Processes lineup information including team compositions and player positions
- Handles missing values and roster changes between seasons

### Feature Engineering
- Team and player encoding for categorical variables
- Player frequency statistics by team and position
- Lineup compatibility features to capture which players work well together
- Temporal features to account for season-specific patterns

### Modeling Strategy
- **Random Forest Classifier**: Primary model with optimized hyperparameters (n_estimators=200, max_depth=15)
- **Team-Specific Models**: Separate models for teams with sufficient data
- **Year-Specific Training**: Uses sliding window approach to select relevant training years
- **Statistical Fallbacks**: Leverages player frequency when models lack confidence

### Post-Processing
- Validation of predictions against team rosters
- Ensemble prediction strategy combining multiple model outputs
- Specialized handling for different seasons, particularly 2016

## Instructions for Setup and Running

### Prerequisites
- Python 3.7 or higher
- Required packages: pandas, numpy, scikit-learn

### Installation
1. Clone this repository to your local machine
2. Install the required packages:
```bash
pip install pandas numpy scikit-learn
```

### Data Structure
Ensure your data is organized as follows:

```
NBALINEUPPREDICTION/
│── dataset/
│   ├── matchups-YEAR.csv
│── test_dataset/
│   ├── NBA_test_labels.csv
│   ├── NBA_test.csv
│── main.py
```

## Running the Code
Execute the main script:

```bash
python nba_prediction.py
```

## Key Findings
1. **Overall Accuracy**  
   The overall accuracy was 73.1% across all years

2. **Overall F1 Score**  
   The overall F1 Score was 73.6%

3. **Best Performing Season**  
   The best performing season was 2008 with an accuracy of 84%
