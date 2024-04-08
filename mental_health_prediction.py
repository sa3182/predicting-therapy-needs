"""
CSCI-635: Final Project
file: mental_health_prediction.py
description: Predicting mental health needs using survey dataset
language: python3
author: Shravani Athkuri, sa3182@rit.edu
"""


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def read_data():
    """
    This method reads the data from csv file and stores it as a data frame
    :return: the data frame
    """
    data = pd.read_csv("survey_data.csv")
    print(data)
    print(data.info())
    return data


def data_preprocessing(data):
    """
    This method pre-process the data
    :return: the data frame after pre processing
    """
    # dropping timestamp, country, state and comments to remove biased results
    data.drop(columns=['Timestamp', 'Country', 'state', 'comments'], inplace=True)

    # printing the unique data in each non-numeric columns
    print("Unique Age Groups")
    print(data['Age'].unique())
    print("Unique Genders")
    print(data['Gender'].unique())
    data.drop(data[data['Age'] < 18].index, inplace=True)
    data.drop(data[data['Age'] > 80].index, inplace=True)
    data['Age'].unique()
    # printing the sum of null values
    print(data.isnull().sum())
    # filling the null values
    data['work_interfere'] = data['work_interfere'].fillna('Unknown')
    print(data['work_interfere'].unique())
    data['self_employed'] = data['self_employed'].fillna('No')
    print(data['self_employed'].unique())
    print(data.isnull().sum())

    # replaces the gender with three categories, male, female and other
    print(data['Gender'].value_counts().reset_index())
    data['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                            'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                            'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make', ], 'Male', inplace=True)

    data['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                            'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                            'woman', ], 'Female', inplace=True)

    data["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                            'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                            'Agender', 'A little about you', 'Nah', 'All',
                            'ostensibly male, unsure what that really means',
                            'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                            'Guy (-ish) ^_^', 'Trans woman', ], 'Other', inplace=True)

    print(data['Gender'].value_counts())

    # non-numerical columns
    nominal_data = ['Gender', 'self_employed', 'family_history', 'treatment',
                    'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                    'benefits', 'care_options', 'wellness_program', 'seek_help',
                    'anonymity', 'leave', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor',
                    'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence']
    # numerical columns
    numerical_columns = ['Age']

    # count of different categories of non-numerical data
    count_based_on_categorical_features(data, nominal_data[:11])
    count_based_on_categorical_features(data, nominal_data[11:])

    # encoding non-numerical data to numerical using label encoding
    label_encoder = LabelEncoder()
    for each_column in nominal_data:
        label_encoder.fit(data[each_column])
        data[each_column] = label_encoder.transform(data[each_column])
    print(data)

    # displaying the correlation of the data
    correlation = data.corr()
    print(correlation)
    figure = plt.figure(figsize=(20, 12))
    sns.heatmap(correlation,
                square=False, linewidths=.9, cbar_kws={"shrink": .9}, annot=True, annot_kws={"size": 8})
    plt.show();
    # Normalization
    data = normalization(data, numerical_columns)
    print(data[numerical_columns])
    # Sampling
    print(data['treatment'].value_counts())
    return data


def normalization(data, numerical_columns):
    """
    performs the normalization of the data. Min-max normalization is performed on the numerical columns of the data frame
    :param data: the data frame on which normalization is performed
    :param numerical_columns: the numerical columns in the data set
    :return: the data frame after the normalization is performed
    """
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data[numerical_columns] = min_max_scaler.fit_transform(data[numerical_columns])
    print(data[numerical_columns])
    return data


def count_based_on_categorical_features(data, categorical_columns):
    """
    This method provides a graph showing count of each category in different categorical columns in the data set
    :param data: the data frame
    :param categorical_columns: the categorical columns
    :return: None
    """
    plt.figure(figsize=(20, 12), facecolor='white')
    plotnum = 1
    for cat in categorical_columns:
        axis = plt.subplot(4, 3, plotnum)
        axis.tick_params(axis='x', rotation=90)
        sns.countplot(x=cat, data=data)
        plt.xlabel(cat)
        plt.title(cat)
        plotnum += 1

    plt.subplots_adjust(hspace=0.8)
    plt.show();


def train_models(data):
    """
    Trains different classification models and posts the results
    :param data: the data frame which is used for training
    :return:None
    """
    X = data.drop('treatment', axis=1)
    y = data['treatment']

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.20,
                                                        random_state=101)

    # Splitting data into train and test
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    #     x_train, x_test, y_train, y_test = oversample_the_data(x, y)
    # train and test datasets dimensions
    print(x_train.shape, x_test.shape)
    models_list = []

    models_list.append(train_gaussianNB(x_train, x_test, y_train, y_test))
    models_list.append(train_decision_tree_classifier(x_train, x_test, y_train, y_test))
    models_list.append(train_random_forest_classifier(x_train, x_test, y_train, y_test))
    models_list.append(train_k_neighbors_classifier(x_train, x_test, y_train, y_test))
    models_list.append(train_logistic_regression(x_train, x_test, y_train, y_test))
    models_list.append(train_ada_boost_classifier(x_train, x_test, y_train, y_test))
    models_list.append(train_xgb_classifier(x_train, x_test, y_train, y_test))
    models = []
    model_data_frame = pd.DataFrame()
    for i in models_list:
        models.append(i[0])
        dictionary = {}
        dictionary['Model'] = i[4]
        dictionary['Accuracy'] = i[1]
        dictionary['Precision'] = i[2]
        dictionary['Recall'] = i[3]
        model_data_frame = model_data_frame._append(dictionary, ignore_index=True)
    print(model_data_frame)

    ax = plt.gca()
    for i in models:
        RocCurveDisplay.from_estimator(i, x_test, y_test, ax=ax)
    plt.show();

    cm = sns.light_palette('seagreen', as_cmap=True)
    s = model_data_frame.style.background_gradient(cmap=cm)
    print(s)

    plt.figure(figsize=(20, 5))
    sns.set(style="whitegrid")
    ax = sns.barplot(y='Accuracy', x='Model', data=model_data_frame)
    plt.show()


def train_decision_tree_classifier(x_train, x_test, y_train, y_test):
    """
    Training the decision tree classifier model
    :param x_train: the feature data set on which model is trained
    :param x_test: the feature data set on which model is tested
    :param y_train: the target data set on which model is tested
    :param y_test: the target data set on which model is tested
    :return: a tuple of the model, model accuracy, model precision, model recall, model name
    """

    deseciontree_model = DecisionTreeClassifier(max_depth=10, random_state=40)
    deseciontree_model.fit(x_train, y_train)
    y_predicted_deseciontree = deseciontree_model.predict(x_test)
    deseciontree_model.score(x_test, y_test)
    dtc_accuracy = metrics.accuracy_score(y_test, y_predicted_deseciontree)
    dtc_precision = metrics.precision_score(y_test, y_predicted_deseciontree)
    dtc_recall = metrics.recall_score(y_test, y_predicted_deseciontree)
    print("Accuracy of the DecisionTree model:", metrics.accuracy_score(y_test, y_predicted_deseciontree))
    print("Precision of the DecisionTree model:", metrics.precision_score(y_test, y_predicted_deseciontree))
    print("Recall of the DecisionTree model:", metrics.recall_score(y_test, y_predicted_deseciontree))

    cm = confusion_matrix(y_test, y_predicted_deseciontree)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("DecisionTree Confusion Matrix")
    plt.show()
    return deseciontree_model, dtc_accuracy, dtc_precision, dtc_recall, "DecisionTree"


def train_logistic_regression(x_train, x_test, y_train, y_test):
    """
    Training the logistic regression model
    :param x_train: the feature data set on which model is trained
    :param x_test: the feature data set on which model is tested
    :param y_train: the target data set on which model is tested
    :param y_test: the target data set on which model is tested
    :return: a tuple of the model, model accuracy, model precision, model recall, model name
    """

    lgr_model = LogisticRegression(C=10, random_state=40)
    lgr_model.fit(x_train, y_train)
    y_predicted_lgr = lgr_model.predict(x_test)
    lgr_model.score(x_test, y_test)
    lgr_accuracy = metrics.accuracy_score(y_test, y_predicted_lgr)
    lgr_precision = metrics.precision_score(y_test, y_predicted_lgr)
    lgr_recall = metrics.recall_score(y_test, y_predicted_lgr)
    print("Accuracy of the LogisticRegression model:", metrics.accuracy_score(y_test, y_predicted_lgr))
    print("Precision of the LogisticRegression model:", metrics.precision_score(y_test, y_predicted_lgr))
    print("Recall of the LogisticRegression model:", metrics.recall_score(y_test, y_predicted_lgr))

    cm = confusion_matrix(y_test, y_predicted_lgr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("LogisticRegression Confusion Matrix")
    plt.show()
    return lgr_model, lgr_accuracy, lgr_precision, lgr_recall, "LogisticRegression"


def train_k_neighbors_classifier(x_train, x_test, y_train, y_test):
    """
    Training the K neighbors classifier model
    :param x_train: the feature data set on which model is trained
    :param x_test: the feature data set on which model is tested
    :param y_train: the target data set on which model is tested
    :param y_test: the target data set on which model is tested
    :return: a tuple of the model, model accuracy, model precision, model recall, model name
    """

    KNN_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    KNN_model.fit(x_train, y_train)
    y_predicted_knn = KNN_model.predict(x_test)
    KNN_model.score(x_test, y_test)
    knn_accuracy = metrics.accuracy_score(y_test, y_predicted_knn)
    knn_precision = metrics.precision_score(y_test, y_predicted_knn)
    knn_recall = metrics.recall_score(y_test, y_predicted_knn)
    print("Accuracy of the KNeighborsClassifier model:", metrics.accuracy_score(y_test, y_predicted_knn))
    print("Precision of the KNeighborsClassifier model:", metrics.precision_score(y_test, y_predicted_knn))
    print("Recall of the KNeighborsClassifier model:", metrics.recall_score(y_test, y_predicted_knn))

    cm = confusion_matrix(y_test, y_predicted_knn)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("KNeighborsClassifier Confusion Matrix")
    plt.show()
    return KNN_model, knn_accuracy, knn_precision, knn_recall, "KNeighborsClassifier"


def train_gaussianNB(x_train, x_test, y_train, y_test):
    """
    Training the Gaussian NB model
    :param x_train: the feature data set on which model is trained
    :param x_test: the feature data set on which model is tested
    :param y_train: the target data set on which model is tested
    :param y_test: the target data set on which model is tested
    :return: a tuple of the model, model accuracy, model precision, model recall, model name
    """

    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train)
    y_predicted_gnb = gnb_model.predict(x_test)
    gnb_model.score(x_test, y_test)
    gnb_accuracy = metrics.accuracy_score(y_test, y_predicted_gnb)
    gnb_precision = metrics.precision_score(y_test, y_predicted_gnb)
    gnb_recall = metrics.recall_score(y_test, y_predicted_gnb)
    print("Accuracy of the GaussianNB model:", metrics.accuracy_score(y_test, y_predicted_gnb))
    print("Precision of the GaussianNB model:", metrics.precision_score(y_test, y_predicted_gnb))
    print("Recall of the GaussianNB model:", metrics.recall_score(y_test, y_predicted_gnb))

    cm = confusion_matrix(y_test, y_predicted_gnb)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("GaussianNB Confusion Matrix")
    plt.show()
    return gnb_model, gnb_accuracy, gnb_precision, gnb_recall, "GaussianNB"


def train_random_forest_classifier(x_train, x_test, y_train, y_test):
    """
    Training the random forest classifier model
    :param x_train: the feature data set on which model is trained
    :param x_test: the feature data set on which model is tested
    :param y_train: the target data set on which model is tested
    :param y_test: the target data set on which model is tested
    :return: a tuple of the model, model accuracy, model precision, model recall, model name
    """

    rf_model = RandomForestClassifier(n_estimators=10)
    rf_model.fit(x_train, y_train)
    y_predicted_rf = rf_model.predict(x_test)
    rf_model.score(x_test, y_test)
    rf_accuracy = metrics.accuracy_score(y_test, y_predicted_rf)
    rf_precision = metrics.precision_score(y_test, y_predicted_rf)
    rf_recall = metrics.recall_score(y_test, y_predicted_rf)
    print("Accuracy of the RandomForestClassifier model:", metrics.accuracy_score(y_test, y_predicted_rf))
    print("Precision of the RandomForestClassifier model:", metrics.precision_score(y_test, y_predicted_rf))
    print("Recall of the RandomForestClassifier model:", metrics.recall_score(y_test, y_predicted_rf))

    cm = confusion_matrix(y_test, y_predicted_rf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("RandomForestClassifier Confusion Matrix")
    plt.show()
    return rf_model, rf_accuracy, rf_precision, rf_recall, "RandomForestClassifier"


# AdaBoostClassifier(), xgb.XGBClassifier(random_state=0, booster="gbtree")


def train_ada_boost_classifier(x_train, x_test, y_train, y_test):
    """
    Training the ada boost classifier model
    :param x_train: the feature data set on which model is trained
    :param x_test: the feature data set on which model is tested
    :param y_train: the target data set on which model is tested
    :param y_test: the target data set on which model is tested
    :return: a tuple of the model, model accuracy, model precision, model recall, model name
    """

    abc_model = AdaBoostClassifier()
    abc_model.fit(x_train, y_train)
    y_predicted_abc = abc_model.predict(x_test)
    abc_model.score(x_test, y_test)
    abc_accuracy = metrics.accuracy_score(y_test, y_predicted_abc)
    abc_precision = metrics.precision_score(y_test, y_predicted_abc)
    abc_recall = metrics.recall_score(y_test, y_predicted_abc)
    print("Accuracy of the AdaBoostClassifier model:", metrics.accuracy_score(y_test, y_predicted_abc))
    print("Precision of the AdaBoostClassifier model:", metrics.precision_score(y_test, y_predicted_abc))
    print("Recall of the AdaBoostClassifier model:", metrics.recall_score(y_test, y_predicted_abc))

    cm = confusion_matrix(y_test, y_predicted_abc)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("AdaBoostClassifier Confusion Matrix")
    plt.show()
    return abc_model, abc_accuracy, abc_precision, abc_recall, "AdaBoostClassifier"


def train_xgb_classifier(x_train, x_test, y_train, y_test):
    """
    Training the xgb classifier model
    :param x_train: the feature data set on which model is trained
    :param x_test: the feature data set on which model is tested
    :param y_train: the target data set on which model is tested
    :param y_test: the target data set on which model is tested
    :return: a tuple of the model, model accuracy, model precision, model recall, model name
    """

    xgb_model = XGBClassifier(random_state=0, booster="gbtree")
    xgb_model.fit(x_train, y_train)
    y_predicted_xgb = xgb_model.predict(x_test)
    xgb_model.score(x_test, y_test)
    xgb_accuracy = metrics.accuracy_score(y_test, y_predicted_xgb)
    xgb_precision = metrics.precision_score(y_test, y_predicted_xgb)
    xgb_recall = metrics.recall_score(y_test, y_predicted_xgb)
    print("Accuracy of the XGBClassifier model:", metrics.accuracy_score(y_test, y_predicted_xgb))
    print("Precision of the XGBClassifier model:", metrics.precision_score(y_test, y_predicted_xgb))
    print("Recall of the XGBClassifier model:", metrics.recall_score(y_test, y_predicted_xgb))

    cm = confusion_matrix(y_test, y_predicted_xgb)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("XGBClassifier Confusion Matrix")
    plt.show()
    return xgb_model, xgb_accuracy, xgb_precision, xgb_recall, "XGBClassifier"


def main():
    # read the data
    data = read_data()
    # pre process the data
    data = data_preprocessing(data)
    # train and test the models with the pre-processed data
    train_models(data)


main()
