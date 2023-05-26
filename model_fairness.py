import dalex as dx
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelFairness():

    def __init__(self, path_to_training=None, path_to_test=None):
        self.path_to_training = path_to_training
        self.path_to_test = path_to_test
        self.train_data = pd.read_csv(self.path_to_training)
        self.test_data = pd.read_csv(self.path_to_test)
        self.target = None
    
    def set_target(self, target):
        self.target = target

    # evaluate a model for fairness given protected fields and privileged fields
    def evaluate_model(
        self,
        categorical_features,
        model=DecisionTreeClassifier(),
        protected_field='Gender',
        priveleged_field='Male',
        id_columns=[]
    ):
        interim_dataset = self.train_data

        for col in id_columns:
            interim_dataset = interim_dataset.drop(columns=col)
        
        X = interim_dataset.drop(columns=self.target)
        y = interim_dataset[self.target].apply(lambda x: 1 if x == 'Y' else 0)

        for feat in categorical_features:
            X = pd.get_dummies(X, columns=[feat])
        
        for col in X.columns:
            X[col] = X[col].apply(lambda x: -9999999 if np.isnan(x) else x)

        clf = model

        clf.fit(X, y)

        exp = dx.Explainer(clf, X, y)

        fobject = exp.model_fairness(protected=interim_dataset[protected_field], privileged=priveleged_field)
        fobject.fairness_check(epsilon = 0.8)
        fobject.plot(type = "metric_scores")

    def compare_models(
        self,
        categorical_features,
        models=[DecisionTreeClassifier(), RandomForestClassifier()],
        protected_field='Gender',
        priveleged_field='Male',
        id_columns=[]
    ):
        interim_dataset = self.train_data

        for col in id_columns:
            interim_dataset = interim_dataset.drop(columns=col)
        
        X = interim_dataset.drop(columns=self.target)
        y = interim_dataset[self.target].apply(lambda x: 1 if x == 'Y' else 0)

        for feat in categorical_features:
            X = pd.get_dummies(X, columns=[feat])
        
        for col in X.columns:
            X[col] = X[col].apply(lambda x: -9999999 if np.isnan(x) else x)

        baseline_clf = DecisionTreeClassifier(random_state=42).fit(X, y)

        exp = dx.Explainer(baseline_clf, X, y)

        fobject = exp.model_fairness(protected = interim_dataset[protected_field], privileged=priveleged_field)

        classifiers = []
        explainers = []
        fobjects = []

        for model in models:

            clf_ = model.fit(X,y)

            classifiers.append(clf_)

            exp_  = dx.Explainer(clf_, X, y, verbose = False)

            explainers.append(clf_)

            fob_ = exp_.model_fairness(protected=interim_dataset[protected_field], privileged=priveleged_field)

            fobjects.append(fob_)

            fob_.plot(type = "metric_scores")
            fob_.plot(type = "fairness_check")
        
        return (classifiers, explainers, fobjects)


if __name__ == "__main__":
    mf = ModelFairness(
        path_to_training='data/dataset_3/loan-train.csv',
        path_to_test='data/dataset_3/loan-test.csv'
    )
    mf.set_target('Loan_Status')
    """
    mf.evaluate_model(
        categorical_features=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],
        model=DecisionTreeClassifier(max_depth=7, random_state=42),
        protected_field='Gender',
        priveleged_field='Male',
        id_columns=['Loan_ID']
    )
    """
    mf.compare_models(
        categorical_features=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],
        models=[
            DecisionTreeClassifier(max_depth=7, random_state=42),
            RandomForestClassifier(max_depth=4, random_state=42),
            LogisticRegression(random_state=42),
        ],
        protected_field='Property_Area',
        priveleged_field='Urban',
        id_columns=['Loan_ID']
    )
