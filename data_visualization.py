import matplotlib.pyplot as plt
import seaborn as sns

# def plot_categorical_data(data):
#     """ Plot bar charts for each categorical variable in the dataset. """
#     categorical_columns = data.select_dtypes(include=['object']).columns
#     plt.figure(figsize=(18, 36))
#     for index, col in enumerate(categorical_columns, start=1):
#         y = data[col].value_counts()
#         plt.subplot(11, 4, index)
#         plt.xticks(rotation=90)
#         sns.barplot(x=y.index, y=y)
#     plt.tight_layout()
#     plt.show()

def plot_correlation_heatmap(data):
    """ Plot a heatmap showing correlations between different features. """
    plt.figure(figsize=(12, 6))
    sns.heatmap(data.corr(), cmap='BrBG', annot=True, fmt='.2f', linewidths=2)
    plt.show()

def plot_categorical_relations(data, x, y, hue):
    """ Create a categorical plot for specific columns. """
    sns.catplot(x=x, y=y, hue=hue, kind="bar", data=data)
    plt.show()
