import argparse
import pprint

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_fairness import DataFairness
from model_fairness import ModelFairness
from model_explainer import ModelExplainer


MODELS = [
    DecisionTreeClassifier(max_depth=7, random_state=42),
    RandomForestClassifier(max_depth=4, random_state=42),
    LogisticRegression(random_state=42),
]


def main(args):
    training_path = args.training_path
    test_path = args.test_path
    categorical_features = args.categorical_features.split(',')
    protected_field = args.protected_field
    privileged_field = args.privileged_field
    id_columns = args.id_columns.split(',')
    demographic_fields = args.demographic_fields.split(',')
    gender_field = args.gender_field
    neighborhood_field = args.neighborhood_field
    education_field = args.education_field
    target = args.target

    print("Arguments passed:")
    print("Training Path:", categorical_features)
    print("Test Path:", protected_field)
    print("Categorical Features:", categorical_features)
    print("Protected Field:", protected_field)
    print("Privileged Field:", privileged_field)
    print("ID Columns:", id_columns)
    print("Demographic Fields:", demographic_fields)
    print("Gender Field:", gender_field)
    print("Neihborhood Field:", neighborhood_field)
    print("Education Field:", education_field)
    print("Target:", target)

    df = DataFairness(
        path_to_training=training_path,
        path_to_test=test_path
    )

    results = df.process(
        demographic_fields=demographic_fields,
        gender_field=gender_field,
        neighborhood_field=neighborhood_field,
        education_field=education_field,
    )
    pprint.pprint(results)

    mf = ModelFairness(
        path_to_training=training_path,
        path_to_test=test_path
    )
    mf.set_target(target)

    mf.compare_models(
        categorical_features=categorical_features,
        models=MODELS,
        protected_field=protected_field,
        priveleged_field=privileged_field,
        id_columns=id_columns
    )

    print ("Here are the models we tried:")
    for i, model in enumerate(MODELS):
        print ("Model number:", i, "-", model)
    best_model = input("Which model is the best?")

    exp = ModelExplainer(
        path_to_training=training_path,
        path_to_test=test_path
    )
    exp.set_target(target)

    exp.generate_explainer(
        categorical_features=categorical_features,
        model=MODELS[int(best_model)],
        id_columns=id_columns,
    )
    shap_values = exp.get_shap_values()

    exp.plot_single_point(35)

    exp.plot_summary()


if __name__ == "__main__":
    # Example usage:
    # python loan_prediction_framework.py --training_path 'data/dataset_3/loan-train.csv' --test_path 'data/dataset_3/loan-test.csv' --categorical_features "Gender,Married,Dependents,Education,Self_Employed,Property_Area" --protected_field "Property_Area" --privileged_field "Urban" --id_columns "Loan_ID" --demographic_fields "Gender,Married,Education,Property_Area" --gender_field "Gender" --neighborhood_field "Property_Area" --education_field "Education" --target "Loan_Status"
    parser = argparse.ArgumentParser(description="Argument Parser Example")

    # Add the command line arguments
    parser.add_argument('--training_path', type=str, required=True,
                        help='Path to training dataset')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--categorical_features', type=str, required=True,
                        help='Comma separated list of categorical features')
    parser.add_argument('--protected_field', type=str, required=True,
                        help='The protected field')
    parser.add_argument('--privileged_field', type=str, required=True,
                        help='The privileged field')
    parser.add_argument('--id_columns', type=str, required=True,
                        help='Comma separated list of ID columns')
    parser.add_argument('--demographic_fields', type=str, required=True,
                        help='Comma separated list of categorical features')
    parser.add_argument('--gender_field', type=str, required=False,
                        help='Column containing gender information')
    parser.add_argument('--neighborhood_field', type=str, required=False,
                        help='Column containing neighborhood information')
    parser.add_argument('--education_field', type=str, required=False,
                        help='Column containing education information')
    parser.add_argument('--target', type=str, required=True,
                        help='Column containing the loan approval status')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)








