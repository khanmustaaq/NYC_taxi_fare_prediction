
---

# NYC Taxi Fare Prediction

## Project Overview
This project aims to predict the fare amount for New York City taxi rides using data from historical taxi trips. The dataset contains information on pickup and dropoff coordinates, passenger counts, and fare amounts, which are used to train machine learning models to make predictions.

## Dataset
The dataset used for this project contains the following columns:
- **fare_amount**: The fare charged for the taxi ride (target variable).
- **pickup_datetime**: The date and time when the taxi ride started.
- **pickup_longitude**: The longitude coordinate of the pickup location.
- **pickup_latitude**: The latitude coordinate of the pickup location.
- **dropoff_longitude**: The longitude coordinate of the dropoff location.
- **dropoff_latitude**: The latitude coordinate of the dropoff location.
- **passenger_count**: The number of passengers in the taxi ride.

### Data Source
The dataset was sourced from the NYC Taxi and Limousine Commission (TLC) database, typically available via platforms like Kaggle. Due to the large size of the dataset, only a small random sample is used for efficient model training.

## Key Steps

1. **Data Preprocessing**:
   - Load the dataset and select the relevant columns for modeling.
   - Sample a fraction of the data to speed up the analysis.
   - Handle missing values, if any, and clean the data.

2. **Exploratory Data Analysis (EDA)**:
   - Use visualizations (e.g., histograms, scatter plots) to understand the distribution of fare amounts and the relationship between fare and other features.
   - Inspect geographical data (pickup and dropoff coordinates) to spot any patterns related to taxi zones in NYC.

3. **Feature Engineering**:
   - Derive new features such as the distance between pickup and dropoff points using the geographical coordinates.
   - Extract date and time-related features from `pickup_datetime`, such as hour, day of the week, and seasonality effects.

4. **Model Training**:
   - Train machine learning models such as Linear Regression, Random Forest, or Gradient Boosting to predict taxi fare amounts.
   - Perform hyperparameter tuning to improve model performance.

5. **Model Evaluation**:
   - Evaluate the models using metrics like Mean Squared Error (MSE) and R-squared.
   - Compare different models to determine which one best predicts the taxi fare.

## How to Run the Project

### Prerequisites
To run the project, you will need the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`

You can install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### Running the Code
1. Clone the repository or download the notebook.
2. Download the dataset and ensure it is available in the appropriate directory or update the code with the correct path.
3. Run the Jupyter Notebook to preprocess the data, perform EDA, and train the models.

## Results
The best-performing model was able to predict NYC taxi fares with reasonable accuracy. The Random Forest model (or another selected model) performed better than a simple linear regression model, with improvements in both MSE and R-squared.

### Future Work
- Incorporate additional features like weather data and traffic information to improve prediction accuracy.
- Use advanced models such as XGBoost or deep learning models to further boost performance.
- Explore the impact of surge pricing or time-based pricing (e.g., rush hour) on fare predictions.

## License
This project is licensed under the MIT License.

---
