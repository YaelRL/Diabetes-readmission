# Diabetes
This is a ML Classification project in python, aimed at identifying readmission within 30 days for Diabetes patients.
The data consists of 10,000 labeled inpatients stays. This is an imbalanced data set, with 11% prevalence for readmissions.
Work process included data reading and preparation, train-test split, exploratory data analysis, pre-processing pipelines, model fitting and evaluation.

Abstract: 

Aim: Readmission of patients within 30 days of discharge is a measure of healthcare quality and expenditure. This study’s goal was to build a prediction model for readmission within 30 days for patients with diabetes. 

Methods: the data set consisted of 70,000 labeled admissions of patients with diabetes. Data was cleaned, split to train and test sets and studied in exploratory data analysis. The data was processed, and machine-learning models were fitted and evaluated using stratified cross-validation. 

Results: readmission prevalence was 9%. Our prediction model achieved an Area Under the Curve (AUC) of 0.65, recall of 0.50 and precision of 0.14 on the test set. models for age groups 0-30, 30-70 and 70+ years old were also fitted. The model for age group 0-30 achieved an AUC of 0.77, the model for age group 30-70 had an AUC of 0.64 and the model for age group 70+ yielded an AUC of 0.61. 

Conclusions: our models were underfitted and demonstrated poor discriminative ability. Better features or better data quality may assist in improving our prediction models.
