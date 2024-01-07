from sklearn import metrics

def evaluate_models(models, X, Y, dataset_type):
    """ Evaluate each model and print its accuracy on the given dataset. """
    for model in models:
        Y_pred = model.predict(X)
        accuracy = metrics.accuracy_score(Y, Y_pred)
        print("Accuracy score of {} on {} dataset: {:.2f}%".format(model.__class__.__name__, dataset_type, accuracy * 100))
        print("Mean Absolute Error: {}", metrics.mean_absolute_error(Y, Y_pred))
        print("Mean Absolute Error: {}", metrics.mean_squared_error(Y, Y_pred))

