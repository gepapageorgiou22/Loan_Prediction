import data_preprocessing as dp
import model_training as mt
import model_evaluation as me
import data_visualization as dv  # Importing the data_visualization module

def main():
    # Load the data
    data = dp.load_data("/Users/gepapageorgiou/Documents/Master/Xeimerino Examino/LoanApprovalPrediction.csv")

    # Preprocess the data
    preprocessed_data = dp.preprocess_data(data)

    # Visualize the data
    dp.visualize_data(data)  # Call the function from the data_visualization module

    # Visualize the proccessed data
    # dv.plot_categorical_data(preprocessed_data)
    dv.plot_correlation_heatmap(preprocessed_data)
    dv.plot_categorical_relations(preprocessed_data, 'Gender', 'Married', 'Loan_Status')


    # Split the data
    X_train, X_test, Y_train, Y_test = dp.split_data(preprocessed_data)

    # Initialize and train models
    models = mt.initialize_models()
    trained_models = mt.train_models(models, X_train, Y_train)

    # Evaluate models
    me.evaluate_models(trained_models, X_train, Y_train, 'train')
    me.evaluate_models(trained_models, X_test, Y_test, 'test')

if __name__ == "__main__":
    main()
