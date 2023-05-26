# FELA: Fair and Explainable Loan Approval Framework
FELA (Fair and Explainable Loan Approval) is a Python-based framework designed to assist machine learning engineers in building fair and explainable models for loan approvals. This framework incorporates fairness metrics, Shapley values, and exploratory data analysis techniques to promote transparency and mitigate bias in loan approval systems.

Loan approval processes are critical and must be fair and unbiased to ensure equal opportunities for all individuals. FELA addresses these concerns by providing a set of functionalities that aid in evaluating and improving the fairness and explainability of loan approval models.


## Key Features
- Fairness Metrics: FELA incorporates various fairness metrics such as disparate impact, statistical parity difference, and equal opportunity difference. These metrics help assess the fairness of loan approval models across different demographic groups.

- Shapley Values: The framework calculates Shapley values for each feature, quantifying their contributions to the model's decision-making process. This enables the identification of features that may lead to biased outcomes and assists in building fairer models.

- Exploratory Data Analysis: FELA facilitates exploratory data analysis to compare dataset distributions with the broader US population. This analysis helps uncover potential biases and disparities in the dataset, allowing for data-driven decision-making.

## Getting Started

To get started with FELA, follow these steps:

- Clone the repository: git clone https://github.com/chirag9127/loan-model-framework.git
- Install the required dependencies: pip install -r requirements.txt
- Run the following command to play around with the example datasets: `python loan_prediction_framework.py --training_path 'data/dataset_3/loan-train.csv' --test_path 'data/dataset_3/loan-test.csv' --categorical_features "Gender,Married,Dependents,Education,Self_Employed,Property_Area" --protected_field "Property_Area" --privileged_field "Urban" --id_columns "Loan_ID" --demographic_fields "Gender,Married,Education,Property_Area" --gender_field "Gender" --neighborhood_field "Property_Area" --education_field "Education" --target "Loan_Status"`

## License
This project is licensed under the MIT License.

## Acknowledgments
Thanks to Prof. Ur on an amazing class and thanks to my wife for putting up with me while I worked on this.

## Contact
For any questions or inquiries, please contact us at cmahapat@chicagobooth.edu

Happy fair and explainable loan approvals with FELA!
