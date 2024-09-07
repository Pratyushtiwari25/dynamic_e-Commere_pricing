
# Dynamic Price Optimization for E-commerce

This project aims to build a machine learning model to predict the optimal discount for products in an e-commerce setting. By leveraging features like customer care calls, customer rating, cost of the product, prior purchases, and weight, the model dynamically suggests the best discount to maximize revenue.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [How to Use](#how-to-use)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

Dynamic price optimization is a crucial strategy for maximizing profit in e-commerce. This project utilizes machine learning, specifically Support Vector Regression (SVR), to predict discounts on products based on multiple features. The model can help businesses optimize prices and boost sales by adjusting discounts dynamically.

## Features

- **Customer Care Calls**: Number of customer care calls received.
- **Customer Rating**: Rating given by customers (1 to 5 scale).
- **Cost of the Product**: The price of the product before discount.
- **Prior Purchases**: The number of previous purchases made by the customer.
- **Weight in Grams**: The weight of the product in grams.
- **Discount Offered**: Target variable representing the discount offered.

## Technologies Used

- **Python**: Programming language for building and deploying the model.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-Learn**: Machine learning library for model building and evaluation.
- **Statsmodels**: Statistical modeling (optional).

## Data Preprocessing

1. **Handling Missing Values**: Any missing values are replaced with `0`. This can be adapted based on the dataset.
2. **Encoding Categorical Variables**: Categorical variables are converted to numerical format using one-hot encoding.
3. **Feature Scaling**: Standardization is applied to ensure that all features are on the same scale.

## Exploratory Data Analysis (EDA)

EDA is performed to understand the relationships between features and the target variable. The following plots are generated:

- **Pair Plot**: Visualizes pairwise relationships between features.
- **Correlation Heatmap**: Shows correlations between features.
- **Histograms**: Displays the distribution of each feature.
- **Boxplot**: Highlights the distribution and outliers in 'Cost of the Product'.
- **Countplot**: Shows frequency distribution for categorical variables like 'Mode_of_Shipment'.

## Model Training

The project uses **Support Vector Regression (SVR)** for predicting discounts. SVR is chosen for its effectiveness in handling both linear and non-linear relationships.

- **Feature Selection**: Key features are selected based on domain knowledge and EDA.
- **Train-Test Split**: The data is split into training (80%) and testing (20%) sets.
- **Feature Scaling**: StandardScaler is used to normalize the features.

## Hyperparameter Tuning

Grid Search is utilized for hyperparameter tuning to find the best combination of hyperparameters:

- **Kernel**: `'linear'`, `'rbf'`, `'poly'`
- **C**: `[0.1, 1, 10]`
- **Epsilon**: `[0.01, 0.1, 0.2]`

## Model Evaluation

- **Mean Squared Error (MSE)**: Evaluates the model's performance.
- **Residual Plot**: Used to check for constant variance of residuals.
- **Mean Absolute Percentage Error (MAPE)**: Measures prediction accuracy.
- **Accuracy**: Computes the model's accuracy based on MAPE.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/dynamic-price-optimization.git
   cd dynamic-price-optimization
   ```

2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script**:
   ```bash
   python dynamic_price_optimization.py
   ```

4. **Predict Discount for New Data**:
   - Modify the `new_data` DataFrame in the code with new feature values and run the script to get predicted discounts.

## Results

The model provides accurate predictions for discounts based on various features, enabling dynamic price optimization. It helps businesses adjust prices in real-time to maximize sales and profitability.

- **Visualizations**: The project includes detailed visualizations for better interpretability.
- **Model Performance**: The SVR model demonstrates high accuracy in predicting discounts.

## Conclusion

This project demonstrates the power of machine learning in optimizing prices for e-commerce platforms. By predicting optimal discounts, businesses can dynamically adjust prices, thereby maximizing revenue and enhancing customer satisfaction.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
