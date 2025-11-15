import pandas as pd
import numpy as np
from datetime import timedelta
# import lightgbm as lgb # Not using LGBM for this stable baseline
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# --- Setup ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Define the 5 target counties
TARGET_COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

# --- Competition Metric ---
def competition_metric(y_true, y_pred):
    """Calculates the official competition metric (50% MAE + 50% RMSE)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return (0.5 * mae) + (0.5 * rmse)

# --- Phase 1: Core Data Processing ---
def process_data(agri_df, kamis_df):
    """
    Cleans, filters, and merges the Agribora and KAMIS datasets into a 
    single weekly panel. This function is based on the starter notebook's logic.
    """
    print("Starting data processing...")
    
    # --- 1. Filter and Clean Agribora Data ---
    agr = agri_df[agri_df["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
    agr["county_norm"] = agr["County"].str.strip()
    agr = agr[agr["county_norm"].isin(TARGET_COUNTIES)].copy()
    agr["Date"] = pd.to_datetime(agr["Date"])
    agr["week_start"] = agr["Date"].dt.to_period("W").apply(lambda p: p.start_time)
    agr["agri_price"] = pd.to_numeric(agr["WholeSale"], errors="coerce")
    
    # Aggregate Agribora to weekly mean
    agr_week = agr.groupby(["county_norm", "week_start"], as_index=False)["agri_price"].mean()

    # --- 2. Filter and Clean KAMIS Data ---
    kamis = kamis_df[kamis_df["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
    kamis["county_norm"] = kamis["County"].str.strip()
    kamis = kamis[kamis["county_norm"].isin(TARGET_COUNTIES)].copy()
    kamis["Date"] = pd.to_datetime(kamis["Date"])
    kamis["week_start"] = kamis["Date"].dt.to_period("W").apply(lambda p: p.start_time)
    kamis["kamis_price"] = pd.to_numeric(kamis["Wholesale"], errors="coerce")

    # Aggregate KAMIS to weekly mean
    kamis_week = kamis.groupby(["county_norm", "week_start"], as_index=False)["kamis_price"].mean()

    # --- 3. Create Continuous KAMIS Panel (Fill Gaps & Smooth) ---
    all_kamis_panels = []
    
    # We need the global min/max dates from KAMIS *before* looping
    global_kamis_min = kamis_week["week_start"].min()
    global_kamis_max = kamis_week["week_start"].max()

    for c in TARGET_COUNTIES:
        sub = kamis_week[kamis_week["county_norm"] == c].copy()

        # Create a full weekly index for the county *for the KAMIS range*
        # We handle Agribora dates later
        if sub.empty:
            # Still create a panel, but it will be all NaN
            print(f"Warning: No KAMIS data for {c}")
            full_weeks = pd.date_range(global_kamis_min, global_kamis_max, freq="W-MON")
        else:
            min_d = sub["week_start"].min()
            max_d = sub["week_start"].max()
            # Ensure panel is continuous from global min to max
            full_weeks = pd.date_range(min(min_d, global_kamis_min), max(max_d, global_kamis_max), freq="W-MON")

        df = pd.DataFrame({"week_start": full_weeks})
        df["county_norm"] = c

        if not sub.empty:
            df = df.merge(sub[["week_start", "kamis_price"]], on="week_start", how="left")
            # Fill missing KAMIS prices and apply smoothing
            df["kamis_price"] = df["kamis_price"].ffill().bfill()
        else:
            df["kamis_price"] = np.nan # No data for this county

        df["kamis_smooth"] = df["kamis_price"].rolling(3, min_periods=1, center=True).mean()
        
        all_kamis_panels.append(df)

    if not all_kamis_panels:
        raise ValueError("No KAMIS data found for any target county.")
        
    kamis_panel = pd.concat(all_kamis_panels, ignore_index=True)

    # --- 4. Merge Agribora and KAMIS Panels ---
    # *** BUG FIX 1: Use an 'outer' merge to keep all dates ***
    # This includes Agribora dates that are *after* KAMIS data ends.
    panel = pd.merge(
        kamis_panel,
        agr_week,
        on=["county_norm", "week_start"],
        how="outer" # <--- THIS IS THE FIX
    )
    
    panel = panel.sort_values(["county_norm", "week_start"]).reset_index(drop=True)
    
    # *** BUG FIX 2: Forward-fill the feature gap ***
    # Fill the 'kamis_smooth' values for the new Agribora-only dates
    # We assume the kamis price stays constant after its last observation
    panel["kamis_smooth"] = panel.groupby("county_norm")["kamis_smooth"].ffill()
    # Also backfill for any early Agribora data before KAMIS started
    panel["kamis_smooth"] = panel.groupby("county_norm")["kamis_smooth"].bfill()
    
    print("Data processing complete.")
    return panel

# --- Phase 2: Feature Engineering ---
def create_features(df):
    """
    Creates new features (lags, rolling stats, date parts) for modeling.
    """
    df = df.sort_values(by=["county_norm", "week_start"]).copy()
    
    # --- 1. Baseline Features (from starter notebook) ---
    for lag in [1, 2, 3, 4]:
        df[f"kamis_smooth_lag{lag}"] = df.groupby("county_norm")["kamis_smooth"].shift(lag)

    # --- 2. New Features (for better models) ---
    
    # Lags of the target variable itself (agri_price)
    # We shift by 2 because we predict 2 weeks ahead (H+1 and H+2)
    # The most recent lag we can use for H+1 is H-1 (lag 2 relative to H+1)
    # For simplicity, we'll create lags 1, 2, 3, 4 for general use
    for lag in [1, 2, 3, 4]:
         df[f"agri_price_lag{lag}"] = df.groupby("county_norm")["agri_price"].shift(lag)

    # Rolling statistics
    for window in [4, 8, 12]:
        df[f"agri_price_roll_mean_{window}"] = df.groupby("county_norm")["agri_price"].shift(1).rolling(window, min_periods=1).mean()
        df[f"agri_price_roll_std_{window}"] = df.groupby("county_norm")["agri_price"].shift(1).rolling(window, min_periods=1).std()

    # Date features
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    
    return df

# --- [DELETED] Phase 3: Validation Framework ---
# We are skipping the complex validation for now to replicate the stable benchmark.

# --- Phase 4: Re-written Forecast (Based on Starter Notebook) ---
def generate_forecast_and_submission(panel, model):
    """
    This is the corrected, stable forecasting loop based on the
    original starter notebook's logic.
    """
    print("\n--- Generating Final Forecast (Stable Starter-Notebook Logic) ---")
    
    # These are the first two weeks we MUST predict
    TARGET_START_DATE = pd.Timestamp("2025-11-24") # Week 48
    TARGET_END_DATE = pd.Timestamp("2025-12-01")   # Week 49
    
    # Get the last date of *any* data in our panel
    global_last_week = panel["week_start"].max()
    print(f"Last data point is: {global_last_week.date()}")
    
    forecast_rows = []

    for c in TARGET_COUNTIES:
        hist = panel[panel["county_norm"] == c].sort_values("week_start")
        if hist.empty:
            print(f"No data for {c}, skipping forecast.")
            continue

        # Get the last 3 *known* smoothed kamis prices
        last3 = hist["kamis_smooth"].tail(3).values
        
        # Handle cases where we have less than 3 weeks of data
        if len(last3) == 1:
            lag1 = lag2 = lag3 = last3[-1]
        elif len(last3) == 2:
            lag1 = last3[-1]
            lag2 = lag3 = last3[-2]
        else:
            lag1 = last3[-1]
            lag2 = last3[-2]
            lag3 = last3[-3]

        current_week = global_last_week

        # This loop recursively predicts one week at a time,
        # feeding the new prediction back in as a feature.
        while current_week < TARGET_END_DATE:
            next_week = current_week + timedelta(days=7)
            
            # Create the feature vector for the next step
            # *** BUG FIX 3: Correct column names for preprocessor ***
            X_h = pd.DataFrame({
                "kamis_smooth": [lag1], # Use last prediction as 'current'
                "kamis_smooth_lag1": [lag1],
                "kamis_smooth_lag2": [lag2],
                "kamis_smooth_lag3": [lag3],
                "county_norm": [c]
            })
            
            # Predict H+1
            pred_h = model.predict(X_h)[0]

            # Store the prediction if it's one of our target weeks
            if next_week in [TARGET_START_DATE, TARGET_END_DATE]:
                print(f"Storing prediction for: {c} @ {next_week.date()}")
                forecast_rows.append({
                    "county": c,
                    "week_start": next_week,
                    "agr_pred": pred_h
                })

            # This is the CRITICAL step that was missing:
            # The new prediction becomes the feature for the *next* loop
            lag3 = lag2
            lag2 = lag1
            lag1 = pred_h # The prediction becomes the new 'lag1'
            
            current_week = next_week

    forecast_df = pd.DataFrame(forecast_rows)
    return forecast_df

# --- Phase 5: Create Submission File ---
def create_submission(forecast_df, sample_sub_df):
    """
    Formats the predictions into the required submission format.
    """
    print("\n--- Creating Submission File ---")
    
    # We only need the predictions for weeks 48 and 49
    target_weeks = [
        pd.Timestamp("2025-11-24"), # Week 48
        pd.Timestamp("2025-12-01")  # Week 49
    ]
    forecast_target = forecast_df[forecast_df["week_start"].isin(target_weeks)].copy()
    
    # Create the ID column
    forecast_target["week"] = forecast_target["week_start"].dt.isocalendar().week.astype(int)
    forecast_target["ID"] = forecast_target["county"] + "_Week_" + forecast_target["week"].astype(str)
    
    # Set the target columns
    forecast_target["Target_RMSE"] = forecast_target["agr_pred"]
    forecast_target["Target_MAE"] = forecast_target["agr_pred"]
    
    submission_df = forecast_target[["ID", "Target_RMSE", "Target_MAE"]]
    
    # Use the sample submission to get all the required rows
    # We merge our predictions, keeping all rows from the sample submission
    # Rows we don't have predictions for (e.g., Week 50, 51) will have NaN
    final_sub = sample_sub_df[['ID']].merge(
        submission_df, 
        on="ID", 
        how="left"
    )
    
    # Fill any missing predictions with 0 (as per sample file)
    final_sub = final_sub.fillna(0)
    
    # Ensure correct dtypes
    final_sub["Target_RMSE"] = final_sub["Target_RMSE"].astype(float)
    final_sub["Target_MAE"] = final_sub["Target_MAE"].astype(float)

    # Save the file
    submission_path = "submission.csv"
    final_sub.to_csv(submission_path, index=False)
    
    print(f"Submission file saved to: {submission_path}")
    print("\nFinal Submission Head:")
    print(final_sub.head(15))
    return final_sub

# --- Main Execution ---
def main():
    try:
        # Load all data
        agri_df = pd.read_csv('agribora_maize_prices.csv')
        kamis_df = pd.read_csv('kamis_maize_prices.csv')
        sample_sub_df = pd.read_csv('SampleSubmission.csv')

        # Phase 1: Process data
        panel = process_data(agri_df, kamis_df)
        
        # Phase 2 & 3: Train final model
        # We are training the simple, stable ElasticNet model from the starter
        # notebook on all available data.
        
        print("\nTraining final ElasticNet model on all data.")
        
        # Create features for the *entire* panel
        full_featured_df = create_features(panel)
        
        # Only train on rows where we have both KAMIS features and Agribora price
        model_train_df = full_featured_df.dropna(subset=[
            "kamis_smooth_lag1", 
            "kamis_smooth_lag2", 
            "kamis_smooth_lag3", 
            "agri_price"
        ])
        
        # Define features and pipeline for ElasticNet
        numeric_features = ["kamis_smooth", "kamis_smooth_lag1", "kamis_smooth_lag2", "kamis_smooth_lag3"]
        categorical_features = ["county_norm"]
        
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ])
        
        best_model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=42))
        ])
        
        X_full = model_train_df
        y_full = model_train_df["agri_price"]
        
        best_model.fit(X_full, y_full)
        
        print("ElasticNet model training complete.")
        
        
        # Phase 4: Generate recursive forecast
        # We pass the main 'panel' (not the training subset)
        # to the forecast function, which finds the last date.
        forecast_df = generate_forecast_and_submission(panel, best_model)
        
        # Phase 5: Create submission file
        create_submission(forecast_df, sample_sub_df)
        
        print("\n--- Process Complete! ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'agribora_maize_prices.csv', 'kamis_maize_prices.csv', and 'SampleSubmission.csv' are uploaded to your Colab environment.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()