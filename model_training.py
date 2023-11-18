from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def initialize_models():
    """ Initialize the machine learning models. """
    knn = KNeighborsClassifier(n_neighbors=3)
    rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
    svc = SVC()
    lc = LogisticRegression()
    return [knn, rfc, svc, lc]

def train_models(models, X_train, Y_train):
    """ Train each model on the training dataset. """
    for model in models:
        model.fit(X_train, Y_train)
    return models
