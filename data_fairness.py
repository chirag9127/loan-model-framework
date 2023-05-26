import pandas as pd
import pprint
from scipy.stats import ks_2samp

GENDER_RATIO = 0.5
RURAL_PERCENTAGE = 0.19
SUBURBAN_PERCENTAGE = 0.69
URBAN_PERCENTAGE = 0.12
GRADUATE_PERCENTAGE = 0.53
NOT_GRADUATE_PERCENTAGE = 0.47

class DataFairness():

    def __init__(self, path_to_training=None, path_to_test=None):
        self.path_to_training = path_to_training
        self.path_to_test = path_to_test
        self.train_data = pd.read_csv(self.path_to_training)
        self.test_data = pd.read_csv(self.path_to_test)

    # Check if the distributions for fields are different in training and test
    # for categorical variables
    def are_distributions_different(self, fields, significance_level=0.05):
        results = {}
        for field in fields:
            dummy_train_df = pd.get_dummies(self.train_data[field])
            dummy_test_df = pd.get_dummies(self.test_data[field])
            for col in dummy_train_df.columns:
                statistic, p_value = ks_2samp(dummy_train_df[col], dummy_test_df[col])
                is_different = p_value < significance_level

                results[f"{field}-{col}-train_test_imbalance"] = is_different

                if is_different:
                    print (f"WARNING: Training-Test imbalance Feature: {field}-{col}, Is Different: {is_different}")

        return results
    
    # Check if the gender distribution is in line with the US population
    def check_gender_distribution(self, field, significance_level=0.05):
        if not field:
            return {}
        results = {}
        dummy_train_df = pd.get_dummies(self.train_data[field])
        dummy_test_df = pd.get_dummies(self.test_data[field])

        for col in dummy_train_df.columns:
            if dummy_train_df[col].mean() > GENDER_RATIO * (1 + significance_level) or dummy_train_df[col].mean() < GENDER_RATIO * (1 - significance_level):
                print (f"WARNING: {field}-{col} distribution is not in line with US population")
                results[f"{field}-{col}-us_population_imbalance"] = True
            else:
                results[f"{field}-{col}"] = False

        return results

    # Check if the neighborhood distribution is in line with the US population
    def check_neighborhood_distribution(self, field, significance_level=0.05):
        if not field:
            return {}
        
        results = {}
        dummy_train_df = pd.get_dummies(self.train_data[field])
        dummy_test_df = pd.get_dummies(self.test_data[field])

        for col in dummy_train_df.columns:
            if col in ['Rural']:
                if dummy_train_df[col].mean() > RURAL_PERCENTAGE * (1 + significance_level) or dummy_train_df[col].mean() < RURAL_PERCENTAGE * (1 - significance_level):
                    print (f"WARNING: {field}-{col} distribution is not in line with US population")
                    results[f"{field}-{col}-us_population_imbalance"] = True
                else:
                    results[f"{field}-{col}"] = False
            elif col in ['Semiurban']:
                if dummy_train_df[col].mean() > SUBURBAN_PERCENTAGE * (1 + significance_level) or dummy_train_df[col].mean() < SUBURBAN_PERCENTAGE * (1 - significance_level):
                    print (f"WARNING: {field}-{col} distribution is not in line with US population")
                    results[f"{field}-{col}-us_population_imbalance"] = True
                else:
                    results[f"{field}-{col}"] = False
            elif col in ['Urban']:
                if dummy_train_df[col].mean() > URBAN_PERCENTAGE * (1 + significance_level) or dummy_train_df[col].mean() < URBAN_PERCENTAGE * (1 - significance_level):
                    print (f"WARNING: {field}-{col} distribution is not in line with US population")
                    results[f"{field}-{col}-us_population_imbalance"] = True
                else:
                    results[f"{field}-{col}"] = False

        return results

    # Check if the education distribution is in line with the US population
    def check_education_distribution(self, field, significance_level=0.05):
        if not field:
            return {}
        
        results = {}
        dummy_train_df = pd.get_dummies(self.train_data[field])
        dummy_test_df = pd.get_dummies(self.test_data[field])

        for col in dummy_train_df.columns:
            if col in ['Graduate']:
                if dummy_train_df[col].mean() > GRADUATE_PERCENTAGE * (1 + significance_level) or dummy_train_df[col].mean() < GRADUATE_PERCENTAGE * (1 - significance_level):
                    print (f"WARNING: {field}-{col} distribution is not in line with US population")
                    results[f"{field}-{col}-us_population_imbalance"] = True
                else:
                    results[f"{field}-{col}"] = False
            elif col in ['Not Graduate']:
                if dummy_train_df[col].mean() > NOT_GRADUATE_PERCENTAGE * (1 + significance_level) or dummy_train_df[col].mean() < NOT_GRADUATE_PERCENTAGE * (1 - significance_level):
                    print (f"WARNING: {field}-{col} distribution is not in line with US population")
                    results[f"{field}-{col}-us_population_imbalance"] = True
                else:
                    results[f"{field}-{col}"] = False

        return results
    

    def process(self, demographic_fields=[], gender_field=None, neighborhood_field=None, education_field=None, significance_level=0.05):
        results = {}
        results.update(self.are_distributions_different(demographic_fields, significance_level))
        results.update(self.check_gender_distribution(gender_field, significance_level))
        results.update(self.check_neighborhood_distribution(neighborhood_field, significance_level))
        results.update(self.check_education_distribution(education_field, significance_level))

        return results


if __name__ == "__main__":
    df = DataFairness(
        path_to_training='data/dataset_3/loan-train.csv',
        path_to_test='data/dataset_3/loan-test.csv'
    )
    results = df.process(
        demographic_fields=['Gender', 'Married', 'Education', 'Property_Area'],
        gender_field='Gender',
        neighborhood_field='Property_Area',
        education_field='Education',
    )
    pprint.pprint(results)
