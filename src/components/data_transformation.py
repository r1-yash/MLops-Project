import sys
import os
import numpy as np
import pandas as pd
import pickle

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_transformer(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ])
 
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target = "math_score"

            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]

            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            preprocessor = self.get_transformer()

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            #np ??
            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test]

            # Save preprocessor
            os.makedirs("artifacts", exist_ok=True)
            with open(self.config.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)