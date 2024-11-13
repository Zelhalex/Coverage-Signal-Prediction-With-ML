import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import pickle
import numpy as np
import seaborn as sns
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
# You don’t need to import all the libraries above, as they cover various machine learning algorithms used for comparison here.

# Title and language selection
language = st.selectbox("Select Language:", ["English", "Indonesian"])

# Define titles and labels for both languages
titles = {
    "English": {
        "main_title": "Visualization of Drivetest Map & Comparison of Machine Learning and Conventional Graphic Plots",
        "train_data": "Select Area Type for Training:",
        "prediction_data": "Select Area Type for Prediction:",
        "ml_algorithm_title": "Machine Learning algorithm that was used:",
        "conventional_model_title": "Conventional model that was used:",
        "select_algorithm": "Select Algorithm:",
        "select_model": "Select Model:",
        "warning_features": "Features needed are: Distance Between Tx and Rx (m), Frequency, Altitude, Tinggi Antena, Elevation Angle, Azimuth offset angle, Horizontal Distance of Rx from Boresight of Tx, Vertical Distance of Rx from Boresight of Tx, Longitude Tx, Latitude Tx, LTERSSI, Qual, Tilting Offset Angle",
        "warning_features_note": "The features name must be the same as the one above, if you have more features other than this, it's not a problem",
        "upload_training_file": "Upload CSV File for Training",
        "upload_prediction_file": "Upload CSV File for Prediction",
        "execute_button": "Execute",
        "metric_display": "This metric displays MAE, RMSE, and R-Squared",
        "comparison_plot": "Comparison plot of ML and conventional model in predicting signal level",
        "metrics": ['Mean Absolute Error', 'Root Mean Squared Error', 'R-squared'],
        "signal_colors": {
            -50: 'red',       # Red for -50 dBm
            -60: 'orange',    # Orange for -60 dBm
            -70: 'yellow',    # Yellow for -70 dBm
            -80: 'green',     # Green for -80 dBm
            -90: 'blue',      # Blue for -90 dBm
            -100: 'indigo',   # Indigo for -100 dBm
            -110: 'violet',   # Violet for -110 dBm
            -120: 'gray'      # Gray for -120 dBm
        }
    },
    "Indonesian": {
        "main_title": "Visualisasi Peta Drivetest & Perbandingan Plot Grafik Machine Learning dan Konvensional",
        "train_data": "Pilih Jenis Area untuk Pelatihan:",
        "prediction_data": "Pilih Jenis Area untuk Prediksi:",
        "ml_algorithm_title": "Algoritma Machine Learning yang digunakan:",
        "conventional_model_title": "Model Konvensional yang digunakan:",
        "select_algorithm": "Pilih Algoritma:",
        "select_model": "Pilih Model:",
        "warning_features": "Fitur yang dibutuhkan adalah: Distance Between Tx and Rx (m), Frequency, Altitude, Tinggi Antena, Elevation Angle, Azimuth offset angle, Horizontal Distance of Rx from Boresight of Tx, Vertical Distance of Rx from Boresight of Tx, Longitude Tx, Latitude Tx, LTERSSI, Qual, Tilting Offset Angle",
        "warning_features_note": "Nama fitur harus sama dengan yang di atas, jika Anda memiliki fitur lebih dari ini, tidak masalah",
        "upload_training_file": "Unggah File CSV untuk Pelatihan",
        "upload_prediction_file": "Unggah File CSV untuk Prediksi",
        "execute_button": "Eksekusi",
        "metric_display": "Metrik ini menampilkan MAE, RMSE, dan R-Squared",
        "comparison_plot": "Plot perbandingan model ML dan konvensional dalam memprediksi level sinyal",
        "metrics": ['Mean Absolute Error', 'Root Mean Squared Error', 'R-squared'],
        "signal_colors": {
            -50: 'merah',     # Red for -50 dBm
            -60: 'oranye',    # Orange for -60 dBm
            -70: 'kuning',    # Yellow for -70 dBm
            -80: 'hijau',     # Green for -80 dBm
            -90: 'biru',      # Blue for -90 dBm
            -100: 'indigo',   # Indigo for -100 dBm
            -110: 'ungu',     # Violet for -110 dBm
            -120: 'abu-abu'   # Gray for -120 dBm
        }
    }
}

# Set language based on selection
current_language = titles[language]

# Display titles and labels
st.title(current_language["main_title"])
Train_data = st.selectbox(current_language["train_data"], ["None", "Urban", "Suburban"])
Prediction_data = st.selectbox(current_language["prediction_data"], ["None", "Urban", "Suburban"])

st.title(current_language["ml_algorithm_title"])
Machine_Learning = st.selectbox(current_language["select_algorithm"], ["Random Forest Regression"])
st.title(current_language["conventional_model_title"])
Conventional = st.selectbox(current_language["select_model"], ["Okumura-Hata & COST 231"])

# Function to check if the prediction is available
def is_prediction_available(train, predict):
    available_combinations = [("Urban", "Urban"), ("Urban", "Suburban")]
    return (train, predict) in available_combinations
# For this combination, I currently have only urban data to predict outcomes for both urban and suburban areas.
# If your code includes additional combinations, you may consider adding them here to expand the model’s capabilities.

if is_prediction_available(Train_data, Prediction_data):
    st.warning(current_language["warning_features"])
    st.warning(current_language["warning_features_note"])
    # Upload CSV file for training
    training_file = st.file_uploader(current_language["upload_training_file"], type=["csv"])
    if training_file is not None:
        training_data = pd.read_csv(training_file)

    # Upload CSV file for prediction
    prediction_file = st.file_uploader(current_language["upload_prediction_file"], type=["csv"])
    if prediction_file is not None:
        prediction_data = pd.read_csv(prediction_file)

    # Execute button
    if st.button(current_language["execute_button"]):
        # If both files are uploaded, proceed with the analysis
        if training_file is not None and prediction_file is not None:
            if Train_data == "Urban" and Prediction_data == "Urban":
                training_data.drop(['Number Sampel', 'SNR', 'Horizontal Distance of Rx from Boresight of Tx','Vertical Distance of Rx from Boresight of Tx', 'Longitude Tx', 'Latitude Tx'], axis=1, inplace=True)
# Above this section, the training_data.drop function is used to remove specific features from your file that you don't want to 
# include in the analysis.

# This section of the dashboard interface covers the entire process from file selection to executing the prediction. 
# Users can upload their data files, configure necessary parameters, and initiate the prediction process. 
# Each step is clearly outlined to ensure a smooth workflow, from choosing the file to running the prediction model.

                # Data Seperation as X and Y
                X = training_data.drop(columns=["Signal Level"]) # X as Features (input)
                y = training_data["Signal Level"] # Y as Labels (output)
                X.info()

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                # Data Scaling (Standardization)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Linear Regression
                model_linear_regression = LinearRegression()
                model_linear_regression.fit(X_train_scaled, y_train)

                # Make predictions on the test set
                y_pred_linear_regression = model_linear_regression.predict(X_test_scaled)

                mse_linear_regression = mean_squared_error(y_test, y_pred_linear_regression)
                mae_linear_regression = mean_absolute_error(y_test, y_pred_linear_regression)
                rmse_linear_regression = np.sqrt(mse_linear_regression)
                r2_linear_regression = r2_score(y_test, y_pred_linear_regression)

                # Lasso Regression
                model_lasso = Lasso()
                model_lasso.fit(X_train_scaled, y_train)

                # Make predictions on the test set
                y_pred_lasso = model_lasso.predict(X_test_scaled)

                mse_lasso = mean_squared_error(y_test, y_pred_lasso)
                mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
                rmse_lasso = np.sqrt(mse_lasso)
                r2_lasso = r2_score(y_test, y_pred_lasso)

                # Ridge Regression
                model_ridge = Ridge()
                model_ridge.fit(X_train_scaled, y_train)

                # Make predictions on the test set
                y_pred_ridge = model_ridge.predict(X_test_scaled)

                mse_ridge = mean_squared_error(y_test, y_pred_ridge)
                mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
                rmse_ridge = np.sqrt(mse_ridge)
                r2_ridge = r2_score(y_test, y_pred_ridge)

                # List to store results
                results = []

                # Append Linear Regression results
                results.append({
                    "Model": "Linear Regression",
                    "MSE": mse_linear_regression,
                    "MAE": mae_linear_regression,
                    "RMSE": rmse_linear_regression,
                    "R2": r2_linear_regression
                })

                # Append Ridge Regression results
                results.append({
                    "Model": "Ridge Regression",
                    "MSE": mse_ridge,
                    "MAE": mae_ridge,
                    "RMSE": rmse_ridge,
                    "R2": r2_ridge
                })

                # Convert results to DataFrame for better comparison
                results_df = pd.DataFrame(results)

                # Print the results DataFrame
                print(results_df)

                # Display the DataFrame in Streamlit
                st.write("## Model Comparison Results")
                st.write(results_df)

                # Plot the results using seaborn for better visualization
                st.write("## Mean Squared Error (MSE) Comparison")
                plt.figure(figsize=(10, 6))
                sns.barplot(x="Model", y="MSE", data=results_df)
                plt.xticks(rotation=90)
                plt.xlabel('Model')
                plt.ylabel('Mean Squared Error (MSE)')
                plt.title('MSE Comparison Across Different Models')
                st.pyplot(plt)
# This is just an example to demonstrate how you can create a table to view metric results.
# After this, you can insert the code for your Machine Learning model for Coverage Signal Prediction.

                #CODE FOR ML PREDICTION

                # Signal color mapping based on dBm values
                signal_colors = current_language["signal_colors"]

                # Create map centered around the mean latitude and longitude
                map_center = [prediction_data['Latitude Rx'].mean(), prediction_data['Longitude Rx'].mean()]
                folium_map = folium.Map(location=map_center, zoom_start=10)

                # Add markers with colored circles representing signal strength
                for index, row in prediction_data.iterrows():
                    # Predict signal level using the model
                    features_array = np.array(row[selected_features]).reshape(1, -1)
                    signal_level = model.predict(features_array)[0]

                    # Get color based on signal level's closest dBm value
                    closest_dBm = min(signal_colors.keys(), key=lambda x: abs(x - signal_level))
                    circle_color = signal_colors[closest_dBm]

                    # Add marker with color representing signal strength
                    folium.CircleMarker(
                        location=[row['Latitude Rx'], row['Longitude Rx']],
                        radius=3,
                        color=circle_color,
                        fill=True,
                        fill_color=circle_color,
                        fill_opacity=0.7,
                        popup=f"Signal Level: {signal_level} dBm"
                    ).add_to(folium_map)

                # Display the map
                st.write("Map with Signal Strength Markers")
                st.write("This map displays drivetest data with signal strength information.")
                folium_static(folium_map)

                # Display legend below the map
                st.markdown("""
                    <style>
                        .legend-title {
                            font-weight: bold;
                            margin-bottom: 5px;
                        }
                        .legend-items {
                            list-style: none;
                            padding-left: 0;
                        }
                        .legend-item {
                            margin-bottom: 5px;
                        }
                        .legend-color {
                            width: 15px;
                            height: 15px;
                            display: inline-block;
                            margin-right: 5px;
                            border-radius: 50%;
                        }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown("<div class='legend-title'>Signal Strength Legend</div>", unsafe_allow_html=True)
                for dBm, color in signal_colors.items():
                    st.markdown(f"<div class='legend-item'><span class='legend-color' style='background-color:{color};'></span>{dBm} dBm</div>", unsafe_allow_html=True)
# If you want to create a map, this code will generate one based on the location from which you collected the data. 
# However, you’ll need certain features to complete it.

# This code is not yet complete, as it currently lacks the implementation of both the machine learning model and the conventional model
# for comparison. Be sure to add these components and thoroughly review all code for completeness and accuracy before running it.