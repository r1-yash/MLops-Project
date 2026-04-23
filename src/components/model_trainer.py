import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    EXTRA_MODELS = True
except:
    EXTRA_MODELS = False

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl") #path where it will be stored


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost": AdaBoostRegressor(),
            }

            if EXTRA_MODELS:
                models["XGBoost"] = XGBRegressor()
                models["CatBoost"] = CatBoostRegressor(verbose=False)

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"]
                },
                "Random Forest": {
                    "n_estimators": [16, 32, 64]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [16, 32]
                },
                "Linear Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [16, 32]
                },
            }

            if EXTRA_MODELS:
                params["XGBoost"] = {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [16, 32]
                }
                params["CatBoost"] = {
                    "depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                    "iterations": [30, 50]
                }

            model_report, best_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model = best_models[best_model_name]
            best_score = model_report[best_model_name]

            logging.info(f"Best Model: {best_model_name}, Score: {best_score}")

            if best_score < 0.6:
                raise CustomException("No good model found", sys)

            save_object(self.config.trained_model_file_path, best_model)

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            return r2

        except Exception as e:
            raise CustomException(e, sys)