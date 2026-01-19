# Sklearn Playground

This repository contains machine learning experiments using scikit-learn.

## Regression

The regression module evaluates several linear models on a housing dataset.

### Model Comparison Results

| Model          | RMSE       | MAE       | R²     |
|----------------|------------|-----------|--------|
| Linear Regression | 176662.9777 | 109105.2419 | 0.7601 |
| Ridge         | 176661.9446 | 109104.6940 | 0.7601 |
| Lasso         | 179250.3921 | 110096.7561 | 0.7530 |
| ElasticNet    | 176952.4513 | 109181.7834 | 0.7593 |

### Model Comparison Results (Cross Validation)

| Model          | RMSE       | MAE       | R²     |
|----------------|------------|-----------|--------|
| Linear Regression | 176662.9777 | 109105.2419 | 0.7601 |
| RidgeCV       | 176652.6966 | 109099.7631 | 0.7601 |
| LassoCV       | 176433.7095 | 109055.4153 | 0.7607 |
| ElasticNet    | 176490.9538 | 109064.2048 | 0.7606 |

### Model Theory

- **Linear Regression**: Fits a linear relationship between features and target by minimizing the sum of squared residuals. Simple and interpretable but prone to overfitting with many features.

- **Ridge Regression**: Adds L2 regularization (penalty on squared coefficients) to linear regression, reducing overfitting by shrinking coefficients towards zero.

- **Lasso Regression**: Uses L1 regularization (penalty on absolute coefficients), which can drive some coefficients to exactly zero, creating sparse models and performing feature selection.

- **ElasticNet**: Combines L1 and L2 regularization, balancing the benefits of both Ridge and Lasso, useful when features are correlated.

*Improvements made: Log-transformed target variable for better normality, added house age and renovation age features, increased max_iter for convergence. R² improved from ~0.69 to ~0.76, RMSE reduced from ~200k to ~176k.*