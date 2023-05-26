import numpy as np
import pandas as pd
import shap

from sklearn.tree import DecisionTreeClassifier

class ModelExplainer():

    def __init__(self, path_to_training=None, path_to_test=None):
        self.path_to_training = path_to_training
        self.path_to_test = path_to_test
        self.train_data = pd.read_csv(self.path_to_training)
        self.test_data = pd.read_csv(self.path_to_test)
        self.explainer = None
        self.shap_values = None
        self.X = None
    
    def set_target(self, target):
        self.target = target

    def generate_explainer(
        self,
        categorical_features,
        model=DecisionTreeClassifier(),
        id_columns=[],
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
        
        self.X = X

        clf = model

        clf.fit(X, y)

        self.explainer = shap.Explainer(clf.predict, X)

    def get_shap_values(self):

        if self.explainer is None:
            print (f"WARNING: Please generate explainer first!!")
            return 
        
        self.shap_values = self.explainer(self.X)

        return self.shap_values
    
    def plot_single_point(self, point_id):
        shap.plots.waterfall(self.shap_values[point_id])
    
    def plot_summary(self):
        shap.summary_plot(self.shap_values, self.X, feature_names=self.X.columns, plot_type="bar")



if __name__ == "__main__":
    exp = ModelExplainer(
        path_to_training='data/dataset_3/loan-train.csv',
        path_to_test='data/dataset_3/loan-test.csv'
    )
    exp.set_target('Loan_Status')

    exp.generate_explainer(
        categorical_features=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],
        model=DecisionTreeClassifier(max_depth=7, random_state=42),
        id_columns=['Loan_ID'],
    )

    shap_values = exp.get_shap_values()

    exp.plot_single_point(35)

    exp.plot_summary()
