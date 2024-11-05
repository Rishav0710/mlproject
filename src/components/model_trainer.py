import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException 
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class model_trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):#Transformmed data
        try:
            logging.info("Split training and test input data")

            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models= {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRFRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "ADABoost Regressor": AdaBoostRegressor()
            }
            param_distributions = {
                    "Random Forest": {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    "Decision Tree": {
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    "Gradient Boosting": {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.05],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    },
                    "Linear Regression": {
                        # Linear regression has few hyperparameters; you may not need RandomizedSearchCV for it.
                    },
                    "K-Neighbors Regressor": {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2]
                    },
                    "XGB Regressor": {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.05],
                        'subsample': [0.8, 0.9, 1.0]
                    },
                    "CatBoosting Regressor": {
                        'iterations': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.05],
                        'depth': [4, 6, 8]
                    },
                    "ADABoost Regressor": {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1.0]
                    }
                }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=param_distributions)
            
            #To get best Model score from dict
            best_model_score=max(sorted(model_report.values()))

            #To get best model name from dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and test testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)

            return r2_square



        except Exception as e:
            raise CustomException(e,sys)