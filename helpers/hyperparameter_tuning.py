import optuna

def get_best_params(X_train, y_train):
    def objective(trial):
        # Define the parameter search space
        param = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.7, 0.9),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.7, 0.9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        }

        # Prepare the dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)

        # Perform cross-validation with early stopping
        cv_results = lgb.cv(
            params=param,
            train_set=train_data,
            nfold=3,
            metrics=["auc"],
            # early_stopping_rounds=50,
            seed=42,
        )

        # Get the best AUC score from cross-validation (mean of validation AUC scores)
        return max(cv_results["valid auc-mean"])

    # Run the optimization process with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Retrieve the best parameters
    best_params = study.best_params

    return best_params

# Retrieve the best parameters
best_params = get_best_params(X_train, y_train)
print("Best parameters found:", best_params)
