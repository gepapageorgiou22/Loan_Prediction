from sklearn import metrics

def evaluate_models(models, X, Y, dataset_type):
    """ Evaluate each model and print its accuracy on the given dataset. """
    for model in models:
        Y_pred = model.predict(X)
        accuracy = metrics.accuracy_score(Y, Y_pred)
        print(f"Accuracy score of {model.__class__.__name__} on {dataset_type} dataset: {accuracy * 100:.2f}%")
