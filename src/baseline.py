from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Function to train the model and return predictions
def base_pred(train, valid, predictors, target, n_jobs=-1, random_state=42, criterion="gini", n_estimators=100, verbose=False):
    """
    Train the model and return predictions.
    Parameters:
        train (pd.DataFrame): The training dataset.
        valid (pd.DataFrame): The validation dataset.
        predictors (list): The predictors to use for training.
        target (str): The target variable to predict.   
        n_jobs (int): The number of jobs to run in parallel. Defaults to -1.
        random_state (int): The random state to use for the model. Defaults to 42.
        criterion (str): The criterion to use for the model. Defaults to "gini".
        n_estimators (int): The number of trees to use in the model. Defaults to 100.
        verbose (bool): Whether to print the model summary. Defaults to False.
    Returns:
        Always prints the classification reports.
        dict: A dictionary containing the predictions for both training and validation sets.
    """
    # Preparing the training and validation sets
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values

    # Initializing the Random Forest classifier
    clf = RandomForestClassifier(n_jobs=n_jobs, 
                                 random_state=random_state,
                                 criterion=criterion,
                                 n_estimators=n_estimators,
                                 verbose=verbose)

    # Fitting the model
    clf.fit(train_X, train_Y)

    # Making predictions
    preds_tr = clf.predict(train_X)
    preds_valid = clf.predict(valid_X)

    # Generating classification reports
    train_report = metrics.classification_report(train_Y, preds_tr, target_names=['Not Survived', 'Survived'])
    valid_report = metrics.classification_report(valid_Y, preds_valid, target_names=['Not Survived', 'Survived'])
    
    # Printing the reports
    print("Training Classification Report:\n", train_report)
    print("Validation Classification Report:\n", valid_report)

    return {
        "train_predictions": preds_tr,
        "valid_predictions": preds_valid
    }