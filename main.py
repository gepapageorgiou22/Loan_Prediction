import data_preprocessing as dp
import model_training as mt
import model_evaluation as me

def main():
    # Preprocess the data
    data = dp.load_and_preprocess_data("LoanApprovalPrediction.csv")
    X_train, X_test, Y_train, Y_test = dp.split_data(data)

    # Train the models
    trained_models = mt.train_models(X_train, Y_train)

    # Evaluate the models
    me.evaluate_models(trained_models, X_train, Y_train, 'train')
    me.evaluate_models(trained_models, X_test, Y_test, 'test')

if __name__ == "__main__":
    main()
