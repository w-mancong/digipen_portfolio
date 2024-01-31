# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:00:22 2024

@author: WOng Man Cong
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

def Predict(gamer_csv_filepath, new_data_csv_filepath):
    # Load data into individual variable
    gamer_data = pd.read_csv(gamer_csv_filepath)
    new_data = pd.read_csv(new_data_csv_filepath)
    
    # Separate features and target variables
    x = gamer_data.drop("ActionLatency", axis=1)
    y = gamer_data["ActionLatency"]
    
    # # Train the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # # make prediction on new data
    new_prediction = model.predict(new_data)
    print(*new_prediction) # printing the predictions on another line
    
# Run the script if called from terminal
if __name__ == "__main__":
    import sys
    Predict(sys.argv[1], sys.argv[2])
    # Predict("Gamer.csv", "NewUnlabeledData.csv")