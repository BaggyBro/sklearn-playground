from dataset import load_and_prepare_data
from model import get_models, get_model_cv

def train_models():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    models = get_models()

    trained_models = {}

    for name,model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained successfully...")

    return trained_models, X_test, y_test

def train_models_CV():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    models = get_model_cv()

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained successfully...")

    return trained_models, X_test, y_test

if __name__ == "__main__":
    train_models()