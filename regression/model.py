from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import yaml 

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_models():
    config = load_config()

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=config["models"]["ridge_alpha"]),
        "Lasso": Lasso(alpha=config["models"]["lasso_alpha"], max_iter=10000),
        "ElasticNet": ElasticNet(alpha=config["models"]["elasticnet_alpha"],
                                 l1_ratio=config["models"]["elasticnet_l1_ratio"], max_iter=10000)
    }

    return models

def get_model_cv():
    
    models = {
        "Linear Regression": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=[0.01, 0.1, 1, 10],
                           cv=5),
        "LassoCV": LassoCV(alphas=[0.001, 0.01, 0.1, 1],
                           cv=5,
                           max_iter=10000),
        "ElasticNet": ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1],
                                   cv=5,
                                   max_iter=10000)
    }
    
    return models