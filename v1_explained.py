# --- This is your #1-ranked script, fully explained ---

# --- Step 1: Import all the tools we need ---
import pandas as pd  # For loading and managing our data tables (like Excel)
import numpy as np   # For mathematical operations
from datetime import timedelta  # For adding/subtracting days/weeks from dates
from sklearn.linear_model import ElasticNet  # This is our actual model (the "brain")
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Tools to clean our data
from sklearn.compose import ColumnTransformer  # A tool to organize our cleaning steps
from sklearn.pipeline import Pipeline  # A tool to build an "assembly line" for our model
from sklearn.impute import SimpleImputer  # A tool to fill in missing values
from sklearn.metrics import mean_absolute_error, mean_squared_error  # For scoring
import warnings  # To hide messy warning messages

# --- Setup ---
warnings.filterwarnings('ignore')  # Hides warnings
pd.set_option('display.max_columns', None)  # Lets us see all columns when we print
pd.set_option('display.width', 1000)

# Define the 5 counties we care about
TARGET_COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

# --- Competition Metric ---
def competition_metric(y_true, y_pred):
    """
    This function just calculates the competition's official score.
    It's 50% MAE (average error) and 50% RMSE (error that punishes big mistakes).
    A lower score is better.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return (0.5 * mae) + (0.5 * rmse)

# --- Phase 1: Core Data Processing (The "Secret Weapon") ---
def process_data(agri_df, kamis_df):
    """
    This is the most important function. It finds all your data and gets
    it ready for the model.
    """
    print("Starting data processing...")
    
    # --- 1. Clean Agribora Data (our target prices) ---
    agr = agri_df[agri_df["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
    agr["county_norm"] = agr["County"].str.strip()  # Clean county names
    agr = agr[agr["county_norm"].isin(TARGET_COUNTIES)].copy() # Keep only the 5 counties
    agr["Date"] = pd.to_datetime(agr["Date"]) # Convert date text to a real date
    agr["week_start"] = agr["Date"].dt.to_period("W").apply(lambda p: p.start_time) # Find the Monday of each week
    agr["agri_price"] = pd.to_numeric(agr["WholeSale"], errors="coerce") # Convert price text to a number
    
    # Get the average price for each county for each week
    agr_week = agr.groupby(["county_norm", "week_start"], as_index=False)["agri_price"].mean()

    # --- 2. Clean KAMIS Data (our "clue" prices) ---
    # We do the exact same cleaning steps for the KAMIS file
    kamis = kamis_df[kamis_df["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
    kamis["county_norm"] = kamis["County"].str.strip()
    kamis = kamis[kamis["county_norm"].isin(TARGET_COUNTIES)].copy()
    kamis["Date"] = pd.to_datetime(kamis["Date"])
    kamis["week_start"] = kamis["Date"].dt.to_period("W").apply(lambda p: p.start_time)
    kamis["kamis_price"] = pd.to_numeric(kamis["Wholesale"], errors="coerce")

    # Get the average KAMIS price for each county for each week
    kamis_week = kamis.groupby(["county_norm", "week_start"], as_index=False)["kamis_price"].mean()

    # --- 3. Create Continuous KAMIS Panel ---
    # This part creates a clean, gap-free timeline of KAMIS prices
    all_kamis_panels = []
    global_kamis_min = kamis_week["week_start"].min() # Find the earliest KAMIS date
    global_kamis_max = kamis_week["week_start"].max() # Find the latest KAMIS date (around July 2025)

    for c in TARGET_COUNTIES:
        # Create a full list of *every single week* from the start to the end
        full_weeks = pd.date_range(global_kamis_min, global_kamis_max, freq="W-MON")
        df = pd.DataFrame({"week_start": full_weeks})
        df["county_norm"] = c

        # Add the real KAMIS prices to this timeline
        sub = kamis_week[kamis_week["county_norm"] == c].copy()
        if not sub.empty:
            df = df.merge(sub[["week_start", "kamis_price"]], on="week_start", how="left")
            # Fill any missing weeks with the price from the week before
            df["kamis_price"] = df["kamis_price"].ffill().bfill() 
        else:
            df["kamis_price"] = np.nan # This county had no KAMIS data

        # "Smooth" the price (average of 3 weeks) to remove noise
        df["kamis_smooth"] = df["kamis_price"].rolling(3, min_periods=1, center=True).mean()
        all_kamis_panels.append(df)
        
    kamis_panel = pd.concat(all_kamis_panels, ignore_index=True)

    # --- 4. Merge Agribora and KAMIS Panels (THE #1 SECRET) ---
    
    # *** THIS IS THE SECRET TO YOUR #1 SCORE ***
    # The `kamis_panel` stops in July 2025.
    # The `agr_week` panel has data until September 2025.
    # `how="outer"` tells pandas to keep ALL rows from BOTH tables.
    # This finds your "missing" 2 months of data (Aug/Sept) that others missed!
    panel = pd.merge(
        kamis_panel,
        agr_week,
        on=["county_norm", "week_start"],
        how="outer" # <--- THIS IS THE WINNING MOVE
    )
    
    panel = panel.sort_values(["county_norm", "week_start"]).reset_index(drop=True)
    
    # *** THIS IS THE SECOND PART OF THE SECRET ***
    # Now that we have the Aug/Sept rows, the "kamis_smooth" column is empty for them.
    # `.ffill()` (forward-fill) copies the last known price (from July)
    # into these empty spots. This gives our model a stable "clue" to use.
    panel["kamis_smooth"] = panel.groupby("county_norm")["kamis_smooth"].ffill()
    panel["kamis_smooth"] = panel.groupby("county_norm")["kamis_smooth"].bfill()
    
    print("Data processing complete.")
    return panel

# --- Phase 2: Feature Engineering ---
def create_features(df):
    """
    This function creates the "clues" (features) our model will use
    to make a prediction.
    """
    df = df.sort_values(by=["county_norm", "week_start"]).copy()
    
    # We create 4 "lag" features.
    # lag1 = What was the kamis_smooth price last week?
    # lag2 = What was it 2 weeks ago?
    # ...and so on. This tells the model about the recent past.
    for lag in [1, 2, 3, 4]:
        df[f"kamis_smooth_lag{lag}"] = df.groupby("county_norm")["kamis_smooth"].shift(lag)

    # We add some simple date features.
    # This helps the model learn if prices are always high in December, for example.
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    
    return df

# --- Phase 4: The Forecasting Loop ---
def generate_forecast_and_submission(panel, model):
    """
    This is the "time machine." It predicts the future one week at a time
    until it reaches the target weeks.
    """
    print("\n--- Generating Final Forecast (Stable Starter-Notebook Logic) ---")
    
    # These are the two weeks we need to submit predictions for
    TARGET_START_DATE = pd.Timestamp("2025-11-24") # Week 48
    TARGET_END_DATE = pd.Timestamp("2025-12-01")   # Week 49
    
    # Find the last date we have any data for (thanks to our fix, this is 2025-09-29)
    global_last_week = panel["week_start"].max()
    print(f"Last data point is: {global_last_week.date()}")
    
    forecast_rows = [] # This will store our final predictions

    for c in TARGET_COUNTIES:
        hist = panel[panel["county_norm"] == c].sort_values("week_start")
        if hist.empty:
            continue

        # Get the last 3 prices we know (from Sept 2025)
        # These will be the first "clues" for our time machine
        last3 = hist["kamis_smooth"].tail(3).values
        if len(last3) == 1:
            lag1 = lag2 = lag3 = last3[-1]
        elif len(last3) == 2:
            lag1 = last3[-1]; lag2 = lag3 = last3[-2]
        else:
            lag1 = last3[-1]; lag2 = last3[-2]; lag3 = last3[-3]

        current_week = global_last_week

        # --- The Recursive Loop ---
        # This is like walking up stairs in the dark.
        # 1. Take a step (predict 1 week).
        # 2. Plant your foot (save the prediction).
        # 3. Use that new step to find the next one (predict week 2).
        
        while current_week < TARGET_END_DATE:
            next_week = current_week + timedelta(days=7)
            
            # Create a "feature vector" (our set of clues) for the *next* week
            # *** THIS IS THE BUG FIX THAT STOPPED THE CRASH ***
            # We use the correct column names that our model was trained on
            X_h = pd.DataFrame({
                "kamis_smooth": [lag1], # The "current" price is our last prediction
                "kamis_smooth_lag1": [lag1], # The "1 week ago" price
                "kamis_smooth_lag2": [lag2], # The "2 weeks ago" price
                "kamis_smooth_lag3": [lag3], # The "3 weeks ago" price
                "county_norm": [c]
            })
            
            # Use the trained model to predict this future week's price
            pred_h = model.predict(X_h)[0]

            # If this is one of the weeks we care about, save the prediction
            if next_week in [TARGET_START_DATE, TARGET_END_DATE]:
                print(f"Storing prediction for: {c} @ {next_week.date()}")
                forecast_rows.append({
                    "county": c, "week_start": next_week, "agr_pred": pred_h
                })

            # --- This is the most important part of the loop ---
            # We update our clues for the *next* loop.
            # The "2 weeks ago" price is now the "3 weeks ago" price.
            # The "1 week ago" price is now the "2 weeks ago" price.
            # Our *new prediction* becomes the "1 week ago" price.
            lag3 = lag2
            lag2 = lag1
            lag1 = pred_h # Our prediction becomes the new "clue"
            
            current_week = next_week # Move time forward one week

    forecast_df = pd.DataFrame(forecast_rows)
    return forecast_df

# --- Phase 5: Create Submission File ---
def create_submission(forecast_df, sample_sub_df):
    """
    This function just formats our predictions into the final CSV file.
    """
    print("\n--- Creating Submission File ---")
    
    # Get just the 10 predictions we made (5 counties * 2 weeks)
    target_weeks = [
        pd.Timestamp("2025-11-24"), # Week 48
        pd.Timestamp("2025-12-01")  # Week 49
    ]
    forecast_target = forecast_df[forecast_df["week_start"].isin(target_weeks)].copy()
    
    # Create the ID column (e.g., "Kiambu_Week_48")
    forecast_target["week"] = forecast_target["week_start"].dt.isocalendar().week.astype(int)
    forecast_target["ID"] = forecast_target["county"] + "_Week_" + forecast_target["week"].astype(str)
    
    # Put our prediction in both target columns
    forecast_target["Target_RMSE"] = forecast_target["agr_pred"]
    forecast_target["Target_MAE"] = forecast_target["agr_pred"]
    
    submission_df = forecast_target[["ID", "Target_RMSE", "Target_MAE"]]
    
    # Merge our predictions with the sample file
    # This makes sure we have all the rows the competition expects
    final_sub = sample_sub_df[['ID']].merge(
        submission_df, 
        on="ID", 
        how="left"
    )
    
    # --- This is why we see zeros ---
    # For any rows we didn't predict (like Week 50), fill them with 0.
    # This is correct. The competition ignores these rows.
    final_sub = final_sub.fillna(0)
    
    final_sub.to_csv("submission.csv", index=False)
    
    print(f"Submission file saved to: submission.csv")
    print("\nFinal Submission Head:")
    print(final_sub.head(15)) # Print the first 15 rows
    return final_sub

# --- Main Execution ---
def main():
    """
    This is the main function that runs all the steps in order.
    """
    try:
        # 1. Load data from the CSV files
        agri_df = pd.read_csv('agribora_maize_prices.csv')
        kamis_df = pd.read_csv('kamis_maize_prices.csv')
        sample_sub_df = pd.read_csv('SampleSubmission.csv')

        # 2. Run our "Secret Weapon" data processing function
        panel = process_data(agri_df, kamis_df)
        
        # 3. Train our Model
        print("\nTraining final ElasticNet model on all data.")
        
        # Create features for all the data we have
        full_featured_df = create_features(panel)
        
        # We can only train on rows where we have *both* clues and an answer
        # So we drop rows with missing prices or missing starting clues
        model_train_df = full_featured_df.dropna(subset=[
            "kamis_smooth_lag1", 
            "kamis_smooth_lag2", 
            "kamis_smooth_lag3", 
            "agri_price"
        ])
        
        # --- This is our "Model Assembly Line" ---
        
        # These are the "clues" we will give the model
        numeric_features = ["kamis_smooth", "kamis_smooth_lag1", "kamis_smooth_lag2", "kamis_smooth_lag3"]
        categorical_features = ["county_norm"] # "Categorical" just means text
        
        # Step A: An assembly line for number clues
        # It fills missing values ("Imputer") then scales them ("Scaler")
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # Step B: An assembly line for text clues
        # It converts county names ("Nairobi") into numbers (0,0,1,0,0)
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        
        # This combines our two assembly lines
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ])
        
        # This is the FINAL assembly line:
        # 1. It runs the "preprocessor" to clean the data
        # 2. It sends the clean data to our "model" (ElasticNet) to be trained
        best_model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=42))
        ])
        
        # "X" is all of our clues (features)
        X_full = model_train_df
        # "y" is the answer (the price) we want the model to learn
        y_full = model_train_df["agri_price"]
        
        # This is the "study" command. It tells the model to learn
        # how to get from X (clues) to y (answer).
        best_model.fit(X_full, y_full)
        
        print("ElasticNet model training complete.")
        
        # 4. Run our "Time Machine" to predict the future
        forecast_df = generate_forecast_and_submission(panel, best_model)
        
        # 5. Create the final submission.csv file
        create_submission(forecast_df, sample_sub_df)
        
        print("\n--- Process Complete! ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'agribora_maize_prices.csv', 'kamis_maize_prices.csv', and 'SampleSubmission.csv' are uploaded to your Colab environment.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# This line just tells Python to start at the `main()` function
if __name__ == "__main__":
    main()