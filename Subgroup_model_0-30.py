from cmath import nan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import sklearn

#statistics
import tableone
from tableone import TableOne

#pre-processing
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

#model training and tuning
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import SMOTEN
from collections import Counter
import pickle
from pickle import TRUE

#model evaluation and interpretation
import graphviz
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# ------data reading and cleaning
df = pd.read_csv('data/diabetic_data.csv')

# replacing null values with np.nan
df.replace(['?', "NULL"], np.nan, inplace=True)

#a function for filtering instances by column value
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


#deleting observations for patients who expired/were discharged to hospice
died = [11, 13, 14, 19, 20, 21]
df = filter_rows_by_values(df, "discharge_disposition_id", died)

#deleting observations for newborns and hospice transfers
newborn = [11, 12, 13, 14, 23, 24, 26]
df = filter_rows_by_values(df, "admission_source_id", newborn)

#keeping only one visit per patient
df = df.drop_duplicates("patient_nbr")
#69,986 unique visits

# deleting features with a high proportion of missing data
df.drop(["weight", "payer_code", "medical_specialty",
        "max_glu_serum"], axis=1, inplace=True)

#race- converting to numeric, unifying small categories:
#1=Caucasian, 2=AfricanAmerican, 3=Other
df["race"] = df["race"].map(
    {"Caucasian": 1, "AfricanAmerican": 2, "Asian": 3, "Hispanic": 3, "Other": 3})

#gender- converting to numeric, imputing 3 missing entries with 1=female, most frequent class
df["gender"] = df["gender"].map(
    {"Male": 0, "Female": 1, "Unknown/Invalid": 0}).astype(int)

#converting age to numeric
age_mapping = {"[0-10)": 1, "[10-20)": 2, "[20-30)": 3,
               "[30-40)": 4, "[40-50)": 5, "[50-60)": 6,
               "[60-70)": 7, "[70-80)": 8, "[80-90)": 9,
               "[90-100)": 10}
df["age"] = df["age"].map(age_mapping)

#converting admission types 4,5,6 and 8 to np.nan
df.loc[(df["admission_type_id"].isin([4, 5, 6, 8])),
       "admission_type_id"] = np.nan
#admission type 7 (trauma center, 18 entries) are converted to 1= emergency
df.loc[df["admission_type_id"] == 7, "admission_type_id"] = 1
#1=Emergency
#2=Urgent
#3=Elective
#10% missing values


#checking consistency between discharge (newborn) and age
#mask=df["discharge_disposition_id"]==10 #newborn
#neonate= df.loc[mask]
#neonate["age"]
#age doesnt match, converting to np.nan
df["discharge_disposition_id"].replace(10, np.nan, inplace=True)
# unifying small categories, creating a 3-category discharge disposition feature:
#1=home
#2=short-term hospitalization
#3=long-term care facility
df.loc[(df["discharge_disposition_id"].isin([5, 10, 18, 25, 26])),
       "discharge_disposition_id"] = np.nan
df.loc[(df["discharge_disposition_id"].isin([6, 7, 8, 12, 16, 17])),
       "discharge_disposition_id"] = 1  # home
df.loc[(df["discharge_disposition_id"].isin([9, 15, 22, 27, 28])),
       "discharge_disposition_id"] = 2  # short-term
df.loc[(df["discharge_disposition_id"].isin([4, 23, 24])),
       "discharge_disposition_id"] = 3  # long-term
#76% discharged home, 14% to short-term facility, 4% to long-term and 6% missing data


# unifying small categories, creating a 3-category admission_source_id feature:
#0= otherwise
#1=from emergency room
#2=by clinic/physician referral
df.loc[(df["admission_source_id"].isin([9, 17, 20, 21])),
       "admission_source_id"] = 0
df.loc[(df["admission_source_id"].isin([1, 2, 3, 8])), "admission_source_id"] = 2
df.loc[(df["admission_source_id"].isin([4, 5, 6, 7, 10, 22, 25])),
       "admission_source_id"] = 1

# number_diagnoses: looks like there was a cap at 9. Thereare 73 observations with more
#  than 9 diagnoses. Uniting them with the most relevant (and most frequent) category- 9 diagnoses
df.loc[df["number_diagnoses"] > 9, 'number_diagnoses'] = 9


# A1Cresult-  82% missing data
df["A1C_abnormal"] = df["A1Cresult"].replace({'None': 0, ">8": 2, ">7": 2, "Norm": 1, np.nan: 0})
#0= missing/not tested
#1= A1C tested and was normal
#2= A1CHb abnormal
#13% have abnormal A1C


# converting diagnosis from strings to numerical categories:
#1=circulatory
#2=respiratory
#3=digestive
#4=Diabetes
#5=injury and poisoning
#6= musculoskeletal
#7= genitourinary
#8= cancer
#9=other

def prepare_diagnosis(diag):
    #converting alphanumeric diagnosis strings to numeric values
    df.loc[df[diag].isna(), diag] = "999"  # replacing NaN with numeric string
    df[diag] = df[diag].astype(str)  # converting object type to string type
    df.loc[df[diag].str.match("^[E-V]"), diag] = "999"  # other
    df.loc[df[diag].str.match("^250.*"), diag] = "250"  # diabets
    # converting to numeric
    df[diag] = pd.to_numeric(df[diag], errors='coerce').convert_dtypes()


def categorize_diagnosis(diag):
   #converting numerical diagnoses to categories 1-9
    if 390 <= diag < 460 or diag == 785:  # circulatory
        return 1
    elif (460 <= diag < 520 or diag == 786):  # respiratory
        return 2
    elif (520 <= diag < 579 or diag == 787):  # digestive
        return 3
    elif diag == 250:  # diabets
        return 4
    elif (800 <= diag < 1000):  # injury
        return 5
    elif (710 <= diag < 740):  # musculoskeletal
        return 6
    elif (580 <= diag < 630 or diag == 788):  # genitourinary
        return 7
    elif (140 <= diag < 240):  # neoplasm
        return 8
    else:
        return 9  # other


# converting alphanumeric strings to numeric values
prepare_diagnosis("diag_1")
prepare_diagnosis("diag_2")
prepare_diagnosis("diag_3")

#creating categorical variables for diagnosis 1, 2 and 3
df["diag1_cat"] = df["diag_1"].apply(categorize_diagnosis)
df["diag2_cat"] = df["diag_2"].apply(categorize_diagnosis)
df["diag3_cat"] = df["diag_3"].apply(categorize_diagnosis)

#creating a binary feature: was diabetes one of the first 3 diagnoses for this admission?
df["diabetes_admission"] = ((df["diag1_cat"] == 4) | (
    df["diag2_cat"] == 4) | (df["diag3_cat"] == 4)).astype(int)
#only 41% of admission have a diabetes diagnosis as one of the 3 first diagnoses

# Diabetes medication - creating aggregative features
meds = df[['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
           'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
           'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
           'insulin', 'glyburide-metformin', 'glipizide-metformin',
           'glimepiride-pioglitazone', 'metformin-rosiglitazone',
           'metformin-pioglitazone']]

#creating a feature for the number of diabetes medications per patient
df["num_diabetes_meds"] = (meds != "No").sum(axis=1)
# 70% have at least one diabetes medication, similar to 76% with diabetesMed=1
#unifying small categories
df["num_diabetes_meds"].replace({4: 3, 5: 3}, inplace=True)

#creating a boolean variable, equals true if at least one diabetes medication dose was increased
df["medsUp"] = (meds == "Up").any(axis=1)
#12% had their diabetes meds increased

# same, for a decrease in diabetes emdication dose
df["medsdown"] = (meds == "Down").any(axis=1)
# 12% had their meds decreased
#55% had no change, consistent with "change" variable

# creating a binary feature for whether this patient is using insulin or not
df["insulin"] = np.where(df["insulin"] == "No", 0, 1)
#51% use insulin

# converting string to numeric: 0=no change in diabetes medication, 1= change was made
df["change"].replace({'No': 0, "Ch": 1}, inplace=True)
# 55% had no change, 45%  had their medication changed

#diabetesMed
# covnerting string to numeric: 0=no , 1= yes, diabetes meds are prescribed
df["diabetesMed"].replace({'No': 0, "Yes": 1}, inplace=True)
# 76% have diabetes medication prescribed

#readmitted
# 59% not readmitted, 32% reamitted >30, 9% readmitted <30
#converting to numeric
readmit_map = {"NO": 0, ">30": 1, "<30": 2}
df["readmitted"] = df["readmitted"].map(readmit_map)

#creating our target feature: readmission within 30 days, yes or no
df["readmit30"] = np.where(df['readmitted'] == 2, 1, 0)

# deleting variables, to avoid data leakage, multicollinearity and noise
df.drop(["patient_nbr", "encounter_id", "readmitted", "diag_1", "diag_2", "diag_3", 'metformin', 'repaglinide',
                        'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
                        'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                        'examide', 'citoglipton', 'glyburide-metformin',
                        'glipizide-metformin', 'glimepiride-pioglitazone',
         'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
         'A1Cresult'], axis=1, inplace=True)

# creating a variable for age group: 1: 0-30, 2: 30-70, 3:70+
df['age_cat'] = np.where(df['age'] <= 3, 1, np.where(df['age'] <= 7, 2, 3))

# creating a df for age group 0-30 yo
df1=df[df['age_cat'] == 1].copy()

# this subgroup has a relatively small data set, and number inpatient is an important and 
# a highly skewed feature. we would need to split the data with stratification
# by the target and also by number-inpatient:

# discretizing number_inpatient, for stratification
df1['inpatient'] = np.where(df1['number_inpatient'] > 0, 1, 0)

#creating a feature for stratification by 2 columns: readmit30 and number_inpatient
#https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
df1['inpatient_readmit'] = df1['inpatient'].astype(str) + df1['readmit30'].astype(str)

#--------bivariate stratified test-train split
np.random.seed(42)

#train-test split stratified by readmit30 and inpatient (inpatient admission last year, yes/no) 
train1, test1 = train_test_split(df1, test_size=0.2, random_state=0, stratify=df1[['inpatient_readmit']])


# dividing the train and test sets to X and y sets
train_set1 = train1.reset_index(drop=True)
train_x = train_set1.drop("readmit30", axis=1)
train_y = train_set1["readmit30"].copy()

test_set1 = test1.reset_index(drop=True)
test_x = test_set1.drop("readmit30", axis=1)
test_y = test_set1["readmit30"].copy()

# verifying that the train and test groups are similar overall

#adding a categorical variable "group" with 2 levels: test and train
test_set1['group'] = 'test'
train_set1['group'] = 'train'

# concatenating test and train
test_train = pd.concat([test_set1, train_set1]).reset_index()

# running statistical tests, comparing test and train sets across all features
mytable = TableOne(test_train, groupby='group', pval=True)
print(mytable.tabulate(tablefmt="github"))
# test and train sets have similar characteristics,
# there were no statistically significant differences
# except for num_medications, diag2_cat, diag3_cat and number_outpatient 
# which will not be used


#---------------------------- EDA1-------------------------
#missing data
train_x.isna().sum()/train_x.shape[0]
#race: 2.5%, admission_type_id: 8.4%, discharge_disposition_id: 4.5%

sns.boxplot(x="readmit30", y='time_in_hospital', data=train_set1)
plt.show()



#corr_matrix = train_set1[["cancer", 'circ', "readmit30"]].corr()
train_x["discharge_disposition_id"].value_counts()
corr_matrix = train_set1.corr()
corr_matrix = train_set1[['readmit30', 'discharge_disposition_id']].corr()
corr_matrix["readmit30"].abs().sort_values(ascending=False)

train_set1["home"] = np.where(train_set1["discharge_disposition_id"] > 1 ,0, 1)
#home is a weaker predictor comapred to discharge

train_set1['in_012plus'] = np.where(train_set1['number_inpatient'] < 2, train_set1['number_inpatient'], 2)

train_set1['cancer'] = np.where(train_set1['diag1_cat'] == 8, 1, 0)
train_set1['circ'] = np.where(train_set1['diag1_cat'] == 1, 1, 0)

pd.crosstab(train_set1['diabetes_admission'], train_set1['readmit30'], normalize='index').round(2)

train_set1['race3'] = np.where(train_set1['race'].isna(), 3, train_set1['race'])
pd.crosstab(train_set1['readmit30'], train_set1['race3'], normalize='columns')


g = sns.FacetGrid(train_set1[['number_inpatient', 'number_outpatient',
                  'number_diagnoses', "time_in_hospital", 'readmit30']], col='readmit30')
g.map(plt.hist, 'number_diagnoses', bins=2)
#g.map_dataframe(sns.histplot)
#sns.displot(train_set1[num_vars], x="number_diagnoses", hue="readmit30", element="step")
plt.show()


#there may be an interaction between race and discharge, but not enough data in a lot of cells
pd.crosstab([train_set1['A1C_abnormal'],
             train_set1['race3']], train_set1['readmit30'],
            rownames=['feat', 'race'], colnames=['readmit30'],
            normalize='index')
pd.crosstab(train_set1['race'], train_set1['readmit30'], normalize='index')


#----------------------------------custom transformers
class InpatientDiscretizer(BaseEstimator, TransformerMixin):
    # bin number_inpatient into 3 bins: 0,1 and 2+ inpatient admission
    # in the past year
  def __init__(self):  # no *args or **kargs
      self = self

  def fit(self, X, y=None):
      return self  # nothing else to do

  def transform(self, X):
      col = np.where(X < 2, X, 2)
      return np.c_[col]

class RecDiagFeatureCreator(BaseEstimator, TransformerMixin):
    #creates a binary feature indicating whether this admission was due to
    # a diagnosis known for recurrent admissions:
    #   diag1 = 1 or 5: circulatory or injury
  def __init__(self):  # no *args or **kargs
      self = self

  def fit(self, X, y=None):
      return self  # nothing else to do

  def transform(self, X):
      col = np.where(np.isin(X, [1, 5, 8]), 1, 0)
      return np.c_[col]

class CancerFeatureCreator(BaseEstimator, TransformerMixin):
    #creates a binary feature indicating whether this admission was due to
    # cancer (a diagnosis known for recurrent admissions, diag1 = 8)
  def __init__(self):  # no *args or **kargs
      self = self

  def fit(self, X, y=None):
      return self  # nothing else to do

  def transform(self, X):
      col = np.where(X == 8, 1, 0)
      return np.c_[col]


class TimeDiscretizer(BaseEstimator, TransformerMixin):
    # discretizes time in hospital into 4 bins: 1-2, 3-4, 4-6, 6+
    # in the past year
  def __init__(self):  # no *args or **kargs
      self = self

  def fit(self, X, y=None):
      return self  # nothing else to do

  def transform(self, X):
      col = np.where(X <= 2, 0, np.where(X <= 4, 1, np.where(X <= 6, 2, 3)))
      return np.c_[col]


class CircFeatureCreator(BaseEstimator, TransformerMixin):
    #creates a binary feature indicating whether this admission was due to
    # a circulatory condition, diag1=1
  def __init__(self):  # no *args or **kargs
      self = self

  def fit(self, X, y=None):
      return self  # nothing else to do

  def transform(self, X):
      col = np.where(X == 1, 1, 0)
      return np.c_[col]


class InjuryFeatureCreator(BaseEstimator, TransformerMixin):
    #creates a binary feature indicating whether this admission was due to
    # a circulatory condition, diag1=1
  def __init__(self):  # no *args or **kargs
      self = self

  def fit(self, X, y=None):
      return self  # nothing else to do

  def transform(self, X):
      col = np.where(X == 5, 1, 0)
      return np.c_[col]


#--------------------------pipeline 1
train_set1 = train1.reset_index(drop=True)
train_x = train_set1.drop("readmit30", axis=1)
train_y = train_set1["readmit30"].copy()

# saving features' indexs for use with ColumnTransformer
num_vars = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
            'number_emergency', 'number_inpatient', 'number_diagnoses', 'num_diabetes_meds']
time_ix, lab_ix, procedures_ix, num_emergency_ix, num_inpatient_ix, num_diagnoses_ix, dm_meds_ix = [
    train_x.columns.get_loc(c) for c in num_vars]

cat_vars = ['race', 'age', 'discharge_disposition_id', 'medsUp', 'medsdown', 'diag1_cat', 'A1C_abnormal', 'insulin']
race_ix, age_ix, discharge_ix, medsup_ix, medsdown_ix, diag1_ix, a1c_ix, insulin_ix = [
    train_x.columns.get_loc(c) for c in cat_vars]

#preprocessing pipelines
discharge_pipeline = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=3)),
                               ('ohe', OneHotEncoder())
                             ])

# creating interaction features for discharge and race: imputing, one-hot encoding,
# generating interactions and choosing the best interaction feature
discharge_race_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value=3)),
    ('ohe', OneHotEncoder()),
    ('interactions', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('select', SelectKBest(chi2, k=1))
])

inpatient_pipeline = Pipeline([('bin', InpatientDiscretizer()),
                               ('ohe', OneHotEncoder())
                               ])


#----------tree-based models
tree_columntrans = ColumnTransformer([
    ('bin_inpatient', InpatientDiscretizer(), [num_inpatient_ix]),
    ('discharge', discharge_pipeline, [discharge_ix, race_ix]),
    ('discharge_race', discharge_race_pipeline, [discharge_ix, race_ix]),
    ('bin_num_emergency', Binarizer(threshold=0.9, copy=False), [num_emergency_ix]),
    ('bin_num_diagnoses', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans'), [num_diagnoses_ix]),
    ('scale', MinMaxScaler(), [dm_meds_ix, lab_ix]),
    ('clm_passer', 'passthrough', [age_ix, a1c_ix, insulin_ix])
],  
)


#---models
trf_model = BalancedRandomForestClassifier(class_weight="balanced_subsample")
#trf_model = EasyEnsembleClassifier()
#trf_model = XGBClassifier(scale_pos_weight=17)


# the following transfromers were tested but not used in the final pipeline:
# trasformers for over and under sampling
#trf_over = SMOTEN(sampling_strategy=0.3)
#trf_under= RandomUnderSampler(sampling_strategy=0.5)

#transformers for interactions and feature selection
# trf_selection = SelectKBest(chi2, k=20)
# trf_interactions = PolynomialFeatures(interaction_only=True, include_bias=False)

tree_pipeline = Pipeline([("columntrans", tree_columntrans),
                          ("model", trf_model),
                         ])
for scoring in ['roc_auc', 'recall', 'precision']:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scores = cross_val_score(tree_pipeline, train_x, train_y, scoring=scoring, cv=cv, n_jobs=-1)
    print('Model ', scoring, " mean=", scores.mean(), "stddev=", scores.std())

#Balanced Random forest auc=0.63, recall=0.56, precision=0.09
#easyEnsemble auc=0.64, recall=0.58, precision=0.09
#xgboost auc=0.60, recall=0.15, precision=0.13

#-----distance-based models
reg_columntrans = ColumnTransformer([
    ('bin_inpatient', inpatient_pipeline, [num_inpatient_ix]),
    ('discharge', discharge_pipeline, [discharge_ix, race_ix]),
    ('interaction', discharge_race_pipeline, [discharge_ix, race_ix]),
    ('bin_num_emergency', Binarizer(threshold=0.9, copy=False), [num_emergency_ix]),
    ('bin_num_diagnoses', KBinsDiscretizer(n_bins=4, encode='onehot', strategy='kmeans'), [num_diagnoses_ix]),
    ('scale', StandardScaler(), [dm_meds_ix, lab_ix]),
    ('ohe', OneHotEncoder(), [age_ix]),
    ('clm_passer', 'passthrough', [a1c_ix, insulin_ix])
],  )


#trf_model = LogisticRegression(class_weight='balanced')
#trf_model = LinearSVC(class_weight='balanced', dual=False)
#trf_model = SVC(class_weight="balanced")
trf_model = KNeighborsClassifier()

reg_pipeline = Pipeline([("columntrans", reg_columntrans),
                         ("model", trf_model),
                         ])

for scoring in ['roc_auc', 'recall', 'precision']:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scores = cross_val_score(reg_pipeline, train_x,
                             train_y, scoring=scoring, cv=cv, n_jobs=-1)
    print('Model ', scoring, " mean=", scores.mean(), "stddev=", scores.std())

# logistic regression: auc=0.65, recall=0.43, precision=0.10 (SD recall=0.15!)
# linearSVC: auc=0.64, recall=0.40, precision: 0.10
# SVC (kernel=rbf) auc=0.64, recall= 0.44, precision=0.12
# SVC (kernel=poly) auc=0.62, recall=0.37, precision=0.10
# (KNN auc=0.57, recall=0 - naive classifier

#---3 best models: EasyEnsemble, logistic regression, SVC(kernel=rbf)

#-----Logistic regression hyper parameter tuning
logreg_pipeline = Pipeline([("columntrans", reg_columntrans),
                            ("model", LogisticRegression(class_weight='balanced')),
                         ])

params = {'model__C': [1,10, 100]}
#params = {'model__penalty':[ 'l1', 'l2'], 'model__solver':['liblinear']}
#params = {'model__penalty':[ 'none', 'l2'], 'model__solver':['lbfgs']}
#params = {'model__penalty': ['l2'], 'model__solver': ['lbfgs', 'liblinear']}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = GridSearchCV(logreg_pipeline, param_grid=params,
                    n_jobs=-1, cv=5, scoring='roc_auc', error_score=0)
# params = {'model__n_estimators': [10, 100], 'model__max_features': [1:5]}
# grid = GridSearchCV(tree_pipeline, params, cv=5, scoring='f1')
grid.fit(train_x, train_y)
print(grid.best_score_)
#auc=0.68
print(grid.best_params_)
#penalty=l2, solver=lbgfs, C=1 (default parameters)

#saving best model, after hp tuning
logreg_pipeline = Pipeline([("columntrans", reg_columntrans),
                            ("model", LogisticRegression(class_weight='balanced')),
                            ])
pickle.dump(logreg_pipeline, open('logreg_model1_2.pkl', 'wb'))

# #----SVC hyperparameter tuning
svc_pipeline = Pipeline([("columntrans", reg_columntrans),
                         ("model", SVC(class_weight="balanced")),
                         ])

# hyper parameter tuning
params = {'model__C': [1, 10, 100]}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = GridSearchCV(svc_pipeline, param_grid=params,
                    n_jobs=-1, cv=cv, scoring='roc_auc', error_score=0)
# params = {'model__n_estimators': [10, 100], 'model__max_features': [1:5]}
# grid = GridSearchCV(tree_pipeline, params, cv=5, scoring='f1')
grid.fit(train_x, train_y)
print(grid.best_score_)
#auc=0.68, C=1
print(grid.best_params_)
#C=1

svc_pipeline = Pipeline([("columntrans", reg_columntrans),
                            ("model", SVC(class_weight='balanced')),
                            ])
pickle.dump(svc_pipeline, open('logregsvc_model1.pkl', 'wb'))

#-------Balanced Random forest hyperparameter tuning

brf_pipeline = Pipeline([("columntrans", tree_columntrans),
                         ("model", BalancedRandomForestClassifier(class_weight="balanced_subsample")),
                         ])

# hyperparameter tuning
params = {'model__max_features': ['sqrt','log2', None], 'model__n_estimators': [5000]}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = GridSearchCV(brf_pipeline, param_grid=params,
                    n_jobs=-1, cv=cv, scoring='roc_auc', error_score=0)
# params = {'model__n_estimators': [10, 100], 'model__max_features': [1:5]}
# grid = GridSearchCV(tree_pipeline, params, cv=5, scoring='f1')
grid.fit(train_x, train_y)
print(grid.best_score_)
#auc=0.71
print(grid.best_params_)
#n_estimators=5000, max_depth=none, max_features=sqrt

#saving the model with best parameters
brf_pipeline = Pipeline([("columntrans", tree_columntrans),
                         ("model", BalancedRandomForestClassifier(
                            n_estimators=5000, class_weight="balanced_subsample")),
                         ])
pickle.dump(brf_pipeline, open('brandom_forest1_model1.pkl', 'wb'))

#-------------------------------------------------------------------------
#------------------------ testing the model on the test set-----------------------

easy_pipeline = Pipeline([("columntrans", tree_columntrans),
                         ("model", EasyEnsembleClassifier())
                        ])
easy_model = easy_pipeline.fit(train_x, train_y)
predictions = easy_model.predict(test_x)
probs = easy_model.predict_proba(test_x)
roc_auc_score(test_y, probs[:, 1])
#AUC=0.67

brf_pipeline = Pipeline([("columntrans", tree_columntrans),
                         ("model", BalancedRandomForestClassifier(
                             class_weight="balanced_subsample"))
                          ])
brf_model = brf_pipeline.fit(train_x, train_y)
predictions = brf_model.predict(test_x)
probs = brf_model.predict_proba(test_x)
roc_auc_score(test_y, probs[:, 1])
#auc=0.76

brf_pipeline = Pipeline([("columntrans", tree_columntrans),
                         ("model", BalancedRandomForestClassifier(class_weight="balanced_subsample")),
                         ])
brf_model = brf_pipeline.fit(train_x, train_y)
predictions = brf_model.predict(test_x)
probs=brf_model.predict_proba(test_x)
roc_auc_score(test_y, probs[:,1])
#auc=0.75

logerg_pipeline = Pipeline([("columntrans", reg_columntrans),
                         ("model", LogisticRegression(class_weight="balanced")),
                         ])
logreg_model = logerg_pipeline.fit(train_x, train_y)
predictions = logreg_model.predict(test_x)
probs = logreg_model.predict_proba(test_x)
roc_auc_score(test_y, probs[:,1])
recall_score(test_y, predictions)
precision_score(test_y, predictions)
#auc=0.77, recall= 0.61, precision=0.16

#results: our best model is a logistic regression model, with 
#auc=0.77, recall= 0.61, precision=0.16 on the test set

# ----testing the general model on subgroup 0-30
num_vars = ['admission_type_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses']
admission_type_ix, time_ix, lab_ix, procedures_ix, meds_ix, num_outpatient_ix, num_emergency_ix, num_inpatient_ix, num_diagnoses_ix = [
    train_x.columns.get_loc(c) for c in num_vars]

cat_vars = ['medsdown', 'diabetesMed', 'race', 'admission_type_id', 'discharge_disposition_id', 'age',
            'diag1_cat', 'admission_source_id', 'insulin']
medsdown_ix, dmed_ix, race_ix, admission_type_ix, discharge_ix, age_ix, diag1_ix, admission_source_ix, insulin_ix = [
    train_x.columns.get_loc(c) for c in cat_vars]

admit_pipeline = Pipeline([("impute_admit", SimpleImputer(strategy="constant", fill_value=3)),
                           ("elective_feat", Binarizer(threshold=2.1))])


reg_columntrans = ColumnTransformer([
    ('bin_time', TimeDiscretizer(), [time_ix]),
    ('bin_inpatient', InpatientDiscretizer(), [num_inpatient_ix]),
    ('bin_num_emergency', Binarizer(threshold=0.9, copy=False), [num_emergency_ix]),
    ('bin_num_diagnoses', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans'), [num_diagnoses_ix]),
    ("impute_binarize_admit", admit_pipeline, [admission_type_ix]),
    ('impute_ohe_discharge', discharge_pipeline, [discharge_ix]),
    #("bin_age", AgeMapper(), [age_ix]),
    ("circulatory", CircFeatureCreator(), [diag1_ix]),
    ('injury', InjuryFeatureCreator(), [diag1_ix]),
    ('clmn_passer', 'passthrough', [dmed_ix, medsdown_ix]),
],
)

trans_scale_pipe = Pipeline([("columntrans", reg_columntrans),
                            ('scale', StandardScaler()),
                             ])
logreg_pipeline = Pipeline([("columntrans", reg_columntrans),
                            ('scale', StandardScaler()),
                            ("model", LogisticRegression(class_weight='balanced')),
                            ])

logreg_model = logreg_pipeline.fit(train_x, train_y)
predictions = logreg_model.predict(test_x)
probs = logreg_model.predict_proba(test_x)
roc_auc_score(test_y, probs[:, 1])
recall_score(test_y, predictions)
precision_score(test_y, predictions)
# general model auc=0.77, recall= 0.48 precision=0.21

#------------------------------------- Visualizing results
cm = metrics.confusion_matrix(test_y, predictions, normalize="true")
cm

#plotting ROC curve
#https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
probs = logreg_model.predict_proba(test_x)
fpr, tpr, threshold = metrics.roc_curve(test_y, probs[:, 1])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.show()

# finding the optimal threshold
#https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))
# best threshold = 0.184, G-mean = 0.982
# plot the roc curve for the model
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic regression')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Optimal threshold')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
plt.show()


