from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from train import train_models, train_models_CV
import numpy as np

def eval_model():
    models, X_test, y_test = train_models()

    results = {}

    for name, model in models.items():
        y_pred_log = model.predict(X_test)
        y_pred = np.exp(y_pred_log)  
        y_actual = np.exp(y_test)    

        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_actual, y_pred)),
            "MAE": mean_absolute_error(y_actual, y_pred),
            "R2": r2_score(y_actual, y_pred)
        }

    return results

def eval_model_CV():
    models, X_test, y_test = train_models_CV()

    results = {}

    for name, model in models.items():
        y_pred_log = model.predict(X_test)
        y_pred = np.exp(y_pred_log)
        y_actual = np.exp(y_test)

        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_actual, y_pred)),
            "MAE": mean_absolute_error(y_actual, y_pred),
            "R2": r2_score(y_actual, y_pred)
        }
    
    return results

if __name__ == "__main__":
    results = eval_model()
    results_cv = eval_model_CV()


    print("\nModel Comparison:\n")
    for model, metrics in results.items():
        print(model)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    print("\nModel Comparison using Cross Validation:\n")
    for model, metrics in results_cv.items():
        print(model)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()