import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

mlflow.set_experiment("Bank_Marketing_Routing_Tuned")

def train_and_track():
    print("Fetching and preparing data...")
    bank = fetch_ucirepo(id=222)
    X, y = bank.data.features, bank.data.targets
    X = pd.get_dummies(X, drop_first=True)
    y = y.iloc[:, 0].map({'yes': 1, 'no': 0})
    
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Scale the data
    print("Scaling the Data")

    scaler = StandardScaler()

    #Fit only on the train data to prevent data leakage and then transform
    X_train_scaled = scaler.fit_transform(X_train)


    X_test_scaled = scaler.transform(X_test)
    
    # 3. Apply SMOTE to training data only
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # 4. Define Models and their Hyperparameter/Regularization Grids
    model_configs = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),

            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None], # Regularization: limits tree growth
                "min_samples_split": [2, 5, 10] # Regularization: forces generalized splits
            }
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', 
                                   random_state=42),

            "params": {
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "reg_alpha": [0, 0.1, 1.0],  # L1 Regularization (Lasso)
                "reg_lambda": [1.0, 5.0, 10.0] # L2 Regularization (Ridge)
            }
        },

        #Removed SVC for reducing Compute
        #"SVM": {
        #    "model": SVC(probability=True, random_state=42),
        #   "params": {
        #        "C": [0.1, 1, 10], # Regularization: Lower C = stronger penalty on complexity
        #        "kernel": ['rbf']
        #    }
        #}

        "LogisticRegression" : {
            "model" : LogisticRegression(penalty="elasticnet",
                                         solver="saga",
                                         max_iter=2000,
                                         l1_ratio=0.5,
                                         random_state=42),

            "params" : {
                "C" : [0.01,0.1,10,100],
                "l1_ratio" : [0.0,0.25,0.5,0.75,1.0]
            }
        },

        "DecisionTrees" : {
            "model" : DecisionTreeClassifier(criterion='log_loss',
                                             random_state=42),

            "params" : {
                "max_depth" : [5,10,15,20,None],
                "min_samples_split" : [2,5,10],
                'min_samples_leaf' : [1,2,4,8]
             }                                 
        }
    }

    best_global_score = 0.0
    best_global_model = None
    best_global_name = ""

    # 5. Tune, Cross-Validate, and Track
    for name, config in model_configs.items():
        with mlflow.start_run(run_name=f"{name}_Tuned"):
            print(f"\nRunning 3-Fold Cross-Validation tuning for {name}...")
            
            # Setup RandomizedSearchCV
            # n_iter=5 tests 5 random combinations to save time. cv=3 uses 3-fold cross validation.
            # n_jobs=-1 tells Python to use all your CPU cores to calculate faster.
            search = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["params"],
                n_iter=5, 
                cv=3, 
                scoring='f1',
                n_jobs=-1, 
                random_state=42
            )
            
            # Fit the search on the balanced data
            search.fit(X_train_balanced, y_train_balanced)
            
            # Extract the best model from the CV search
            best_tuned_model = search.best_estimator_
            
            # Test the tuned model on the untouched test set
            preds = best_tuned_model.predict(X_test_scaled)
            score = f1_score(y_test, preds)

            print(f"Best Params for {name}: {search.best_params_}")
            print(f"{name} Final Test F1-Score: {score:.4f}")

            # Log to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_params(search.best_params_) # Logs the winning hyperparameters
            mlflow.log_metric("f1_score", score)
            mlflow.sklearn.log_model(sk_model=best_tuned_model, 
                                     artifact_path=name,
                                     serialization_format="cloudpickle"
                                     )

            # Route the ultimate winner to production
            if score > best_global_score:
                best_global_score = score
                best_global_model = best_tuned_model
                best_global_name = name

    # 6. Export Champion
    os.makedirs("models", exist_ok=True)
    artifact_data = {"model": best_global_model, 
                     "scaler" : scaler,
                     "columns": X.columns.tolist()}
    

    joblib.dump(artifact_data, "models/best_model.pkl")
    print(f"\n🏆 Ultimate Champion: {best_global_name} (F1: {best_global_score:.4f}) saved to disk.")

if __name__ == "__main__":
    train_and_track()