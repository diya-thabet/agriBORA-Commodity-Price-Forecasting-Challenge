# **Zindi AgriBORA Maize Price Prediction**

This is a winning solution for the Zindi "Maize Price Prediction Challenge," designed to forecast weekly maize prices in Kenya.

This script achieved the \#1 rank on the public leaderboard (as of Nov 15, 2025\) by correcting a data processing error in the original starter notebook, which allowed it to train on two extra months of recent data.

## **Competition Goal**

The goal is to predict the average weekly price of dry white maize for five counties in Kenya: Kiambu, Kirinyaga, Mombasa, Nairobi, and Uasin-Gishu. The competition features a rolling leaderboard, where new data is released weekly.

## **How it Works**

The model (ElasticNet) is trained on two datasets. The key to this script's success is in process\_data():

1. It merges agribora\_maize\_prices.csv (which has recent data up to Sept 2025\) and kamis\_maize\_prices.csv (which stops in July 2025).  
2. It uses an how="outer" merge to **keep the "missing" 2 months of data** from the Agribora file that other solutions might miss.  
3. It forward-fills the feature columns to create a complete, recent training set.  
4. It trains a stable ElasticNet model.  
5. It recursively forecasts prices week-by-week to bridge the 7-week gap between the last known data (Sept 29\) and the first prediction target (Nov 24).

## **How to Run**

1. Place the following files in the same directory:  
   * v1\_explained.py (or your script name)  
   * agribora\_maize\_prices.csv  
   * kamis\_maize\_prices.csv  
   * SampleSubmission.csv  
2. Run the script:  
   python v1\_explained.py

3. The script will generate a submission.csv file, which is ready to be uploaded to Zindi.
