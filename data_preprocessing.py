import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """ Load the dataset from a file. """
    data = pd.read_csv(file_path)

    data.head(5) # View the impoered data
    return data

def preprocess_data(data):
    """ Preprocess the dataset: drop unnecessary columns, handle missing values, encode categorical variables. """
    obj = (data.dtypes == 'object')
    print("Categorical variables:",len(list(obj[obj].index))) # Count variables (Categorical variables)

    # Dropping Loan_ID column
    data = data.drop(['Loan_ID'], axis=1)
    visualize_localData(data)
    ## Encoding categorical variables
    label_encoder = preprocessing.LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col].fillna(''))

    # Handling missing numerical values
    for col in data.columns:
        data[col] = data[col].fillna(data[col].mean())

    return data

def visualize_localData(data):
    obj = (data.dtypes == 'object')
    object_cols = list(obj[obj].index)
    plt.figure(figsize=(18,36))
    index = 1

    for col in object_cols:
        y = data[col].value_counts()
        plt.subplot(11,4,index)
        plt.xticks(rotation=90)
        sns.barplot(x=list(y.index), y=y)
        index +=1

    plt.show();

def visualize_data(data):
    """ Plot bar charts for each categorical variable in the dataset showing the distribution of unique values. """
    object_cols = data.select_dtypes(include=['object']).columns
    plt.figure(figsize=(18, 36))
    index = 1
    for col in object_cols:
        y = data[col].value_counts()
        plt.subplot(11,4,index)
        plt.xticks(rotation=90)
        sns.barplot(x=list(y.index), y=y)
        index +=1
    plt.tight_layout()
    plt.show()

def split_data(data):
    """ Split the dataset into training and testing sets. """
    X = data.drop('Loan_Status', axis=1)
    Y = data['Loan_Status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    return X_train, X_test, Y_train, Y_test
