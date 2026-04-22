import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():

            gs = GridSearchCV(model, param[model_name], cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score
            best_models[model_name] = best_model

        return report, best_models   

    except Exception as e:
        raise CustomException(e, sys)