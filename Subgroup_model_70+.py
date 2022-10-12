from cmath import nan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import sklearn

#statistics
# from scipy.stats import chisquare
# from scipy import stats
# from scipy.stats import chi2_contingency
# from scipy.stats import chi2
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
import imblearn
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
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# ------data reading and cleaning
df = pd.read_csv('data/diabetic_data.csv')

# replacing null values with np.nan
df.replace(['?', "NULL"], np.nan, inplace=True)

#a function for filtering instances by column value
#https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value


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
df["A1C_abnormal"] = df["A1Cresult"].replace(
    {'None': 0, ">8": 2, ">7": 2, "Norm": 1, np.nan: 0})
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

# creating a df for age group 70+
df3 = df[df['age_cat'] == 3].copy() #ages 70+


#--------stratified test-train split
np.random.seed(42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df3, df3["readmit30"]):
    train3 = df3.iloc[train_index]
    test3 = df3.iloc[test_index]

# dividing the train and test sets to X and y sets
train_set3 = train3.reset_index(drop=True)
train_x = train_set3.drop("readmit30", axis=1)
train_y = train_set3["readmit30"].copy()

test_set3 = test3.reset_index(drop=True)
test_x = test_set3.drop("readmit30", axis=1)
test_y = test_set3["readmit30"].copy()

# verifying that the train and test groups are similar overall

#adding a categorical variable "group" with 2 levels: test and train
test_set3['group'] = 'test'
train_set3['group'] = 'train'

# concatenating test and train
test_train = pd.concat([test_set3, train_set3]).reset_index()

# running statistical tests, comparing test and train sets across all features
mytable = TableOne(test_train, groupby='group', pval=True)
print(mytable.tabulate(tablefmt="github"))
# test and train sets have similar characteristics,
# there were no statistically significant differences
# (except for number_emergency which was not used)

#---------------------------- EDA3-------------------------
train_set3 = train3.reset_index(drop=True)
train_x = train_set3.drop("readmit30", axis=1)
train_y = train_set3["readmit30"].copy()
# positive class-  10% 

# testing correlations between features and the target
corr_matrix = train_set3.corr()
corr_matrix["readmit30"].abs().sort_values(ascending=False)

#plotting distributions

#sns.boxplot(x="readmit30", y='time_in_hospital', data=train_set3)
sns.boxplot(x="discharge_disposition_id", y='time_in_hospital', data=train_set3)
plt.show()

g = sns.FacetGrid(train_set3, col='race', hue="readmit30")
g.map(sns.histplot, 'time_in_hospital')
plt.show()

sns.catplot(data=train_set3, x='readmit30', y='time_in_hospital',
    col='race', kind='box', col_wrap=2)
plt.show()

sns.catplot(data=train_set3, x='readmit30', y='time_in_hospital',
            col='discharge_disposition_id', kind='box', col_wrap=2)
plt.show()

pd.crosstab([train_set3['discharge_disposition_id'],
        train_set3['race']], train_set3['readmit30'],
    rownames=['discharge_disposition_id','race'], colnames=['readmit30'],
normalize='index')
# interaction between race and discharge disposition, for readmit30 

# testing imputation strategies for discharge disposition
#imputing with 4th category
train_set3["discharge4"] = np.where(train_set3["discharge_disposition_id"].isna(), 4, train_set3["discharge_disposition_id"])
#imputing with most frequent category
train_set3["discharge1"] = np.where(train_set3["discharge_disposition_id"].isna(), 1, train_set3["discharge_disposition_id"])
# imputing with 3=long term facility
train_set3["discharge3"] = np.where(train_set3["discharge_disposition_id"].isna(), 3, train_set3["discharge_disposition_id"])
# testing the correlation between the target and all possible imputation strategies
corr_matrix = train_set3[['discharge_disposition_id', 'discharge4', 'discharge1', 'discharge3','readmit30']].corr()
corr_matrix["readmit30"].abs().sort_values(ascending=False)

# testing discretization strategies

#number_inpatient
# discretizing to 0,1 and 2+ inpatient admissions in the past year
train_set3['in_012plus'] = np.where(train_set3['number_inpatient'] < 2, train_set3['number_inpatient'], 3)
train_set3['in_01plus']=np.where(train_set3['number_inpatient'] == 0, 0, 1)

pd.crosstab(train_set3['in_012plus'], train_set3['readmit30'], normalize='index')
corr_matrix = train_set3[['in_012plus', 'in_01plus', 'number_in[atient','readmit30']].corr()
corr_matrix["readmit30"].abs().sort_values(ascending=False)

#num_diagnoses
train_set3["num_diag"] = np.where(train_set3['number_diagnoses'] <= 3, 0, 1)
corr_matrix = train_set3[['number_diagnoses', 'num_diag','readmit30']].corr()
corr_matrix["readmit30"].abs().sort_values(ascending=False)
#discretizing into 3 bins: 0, 1,2+ is better than the original feature

#time_in_hospital
train_set3['time246'] = np.where(train_set3['time_in_hospital'] <= 2, 0, 
                        np.where(train_set3['time_in_hospital'] <= 4, 1, 
                        np.where(train_set3['time_in_hospital'] <= 6, 2,3)))
train_set3['time35'] = np.where(train_set3['time_in_hospital'] <= 3, 0, 
                    np.where(train_set3['time_in_hospital'] <= 5, 1, 2))
train_set3['time36'] = np.where(train_set3['time_in_hospital']<=3, 0, np.where(train_set3['time_in_hospital']<=6, 1,2))
train_set3['time246'].value_counts()
pd.crosstab(train_set3['time246'], train_set3['readmit30'], normalize='index')


corr_matrix = train_set3.corr()
corr_matrix["readmit30"].abs().sort_values(ascending=False)
#time 246 has the best correlation to readmit30, better than the original feature

g = sns.FacetGrid(train_set3[['number_inpatient', 'number_outpatient',
                  'number_diagnoses', "time_in_hospital", 'readmit30']], col='readmit30')
g.map(plt.hist, 'number_diagnoses', bins=1)
#g.map_dataframe(sns.histplot)
#sns.displot(df[num_vars], x="number_diagnoses", hue="readmit30", element="step")
plt.show()

# testing new features
train_set3['injury']=np.where(train_set3['diag1_cat'] == 5, 1,0)
train_set3['emergency'] = np.where(train_set3['number_emergency'] >0, 1, 0)
corr_matrix = train_set3[['emergency','number_emergency', 'readmit30']].corr()
corr_matrix["readmit30"].abs().sort_values(ascending=False)
#weak correlation

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


class TimeDiscretizer(BaseEstimator, TransformerMixin):
    # bin time in hospital into 4 bins: 1-2, 3-4, 4-6, 6+ days
  def __init__(self):  # no *args or **kargs
      self = self

  def fit(self, X, y=None):
      return self  # nothing else to do

  def transform(self, X):
      col = np.where(X <= 2, 0, np.where(X <= 4, 1, np.where(X <= 6, 2, 3)))
      return np.c_[col]


class InjuryFeatureCreator(BaseEstimator, TransformerMixin):
    #creates a binary feature indicating whether this admission was due to
    # an injury/fall, diag1 = 5
  def __init__(self):  # no *args or **kargs
      self = self

  def fit(self, X, y=None):
      return self  # nothing else to do

  def transform(self, X):
      col = np.where(X == 5, 1, 0)
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




#-----------------------------------pipeline3
# saving features' indexs for use with ColumnTransformer
num_vars = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses']
time_ix, lab_ix, procedures_ix, meds_ix, num_outpatient_ix, num_emergency_ix, num_inpatient_ix, num_diagnoses_ix = [
    train_x.columns.get_loc(c) for c in num_vars]

cat_vars = ['medsdown','diabetesMed','race', 'admission_type_id', 'discharge_disposition_id', 'age',
            'diag1_cat', 'admission_source_id', 'insulin']
medsdown_ix, dmed_ix, race_ix, admission_type_ix, discharge_ix, age_ix, diag1_ix, admission_source_ix, insulin_ix = [
    train_x.columns.get_loc(c) for c in cat_vars]

#preprocessing pipelines
discharge_pipeline = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=3)),
                               ('ohe', OneHotEncoder())
                               ])

discharge_race_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value=3)),
    ('ohe', OneHotEncoder()),
    ('interactions', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('select', SelectKBest(chi2, k=3))
                               ])

#----tree-based models
tree_columntrans = ColumnTransformer([
    ('bin_inpatient', InpatientDiscretizer(), [num_inpatient_ix]),
    ('bin_time', TimeDiscretizer(), [time_ix]),
    ('discharge_race', discharge_race_pipeline, [race_ix, discharge_ix]),
    ('bin_num_diag', Binarizer(threshold=3.1, copy=False), [num_diagnoses_ix]),
    ('injury', InjuryFeatureCreator(), [diag1_ix]),
    ('clmn_passer', 'passthrough', [dmed_ix, medsdown_ix]),
],  
)
# the following transfromers were tested but not used in the final pipeline:
# trasformers for over and under sampling
#trf_over = SMOTEN(sampling_strategy=0.3)
#trf_under= RandomUnderSampler(sampling_strategy=0.5)

#transformers for interactions and feature selection
# trf_selection = SelectKBest(chi2, k=20)
# trf_interactions = PolynomialFeatures(interaction_only=True, include_bias=False)

# tree-based models:
trf_model = BalancedRandomForestClassifier(class_weight='balanced_subsample')
#trf_model = XGBClassifier(scale_pos_weight=10)
#trf_model = EasyEnsembleClassifier()

tree_pipeline = Pipeline([("columntrans", tree_columntrans),
                          ("model", trf_model),
                         ])

# fitting and evaluating model's performance with statified 5-fold cross-vlaidation
for scoring in ['roc_auc', 'recall', 'precision']: 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scores = cross_val_score(tree_pipeline, train_x, train_y, scoring=scoring, cv=cv, n_jobs=-1)
    print("Model ", scoring, " mean=", scores.mean(), "stddev=", scores.std())
# balanced RF auc=0.61, recall=0.53, precision=0.15
# xgboost auc=0.60, recall= 0.52, precision=0.13
# easyensemble auc=0.62, recall=0.52, precision=0.15 

#------distance-based models
reg_columntrans = ColumnTransformer([
    ('bin_inpatient', InpatientDiscretizer(), [num_inpatient_ix]),
    ('bin_time', TimeDiscretizer(), [time_ix]),
    ('discharge_race', discharge_race_pipeline, [race_ix, discharge_ix]),
    ('bin_num_diag', Binarizer(threshold=3.1, copy=False), [num_diagnoses_ix]),
    ('injury', InjuryFeatureCreator(), [diag1_ix]),
    ('clmn_passer', 'passthrough', [dmed_ix, medsdown_ix]),
], 
)

# models:
trf_model = LogisticRegression(class_weight="balanced")
#trf_model = LinearSVC(class_weight='balanced', dual=False)
#trf_model = SVC(class_weight="balanced") #rbf kernel
#trf_model = SVC(class_weight="balanced", kernel='poly')
#trf_model = KNeighborsClassifier(n_neighbors=20)

reg_pipeline = Pipeline([('trf_columntrans', reg_columntrans),
                         ('model', trf_model)
                         ])


for scoring in ['roc_auc', 'recall', 'precision']:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scores = cross_val_score(reg_pipeline, train_x, train_y, scoring=scoring, cv=cv, n_jobs=-1)
    print("Model ", scoring, " mean=", scores.mean(), "stddev=", scores.std())
# auc logistic regression 0.62, recall=0.53, precision=0.15
#auc SVC (kernel rbf)=0.60, recall=0.49 , precision=0.14, 0.6 with interactions
#auc SVC (kernel poly)=0.60, recall=0.50 , precision=0.14
#auc knn = 0.58 recall and precision=0

# 2 best models are: Logistic regression and EasyEnsemble 
# (auc=0.62, recall=0.52-0.53, precision=0.15)


#--------------------hyperparameter tuning
#-----------------------Logistic regression
logreg_pipeline = Pipeline([("columntrans", reg_columntrans),
                            ("model", LogisticRegression(class_weight='balanced')),
                            ])

# using stratified cross-validation grid search for hp tuning
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
#params = {'model__C': [2,5,10], 'model__penalty':[ 'l2', 'none']}
params = {'model__C': [1,5,10]}
#params = {'model__penalty':[ 'none', 'l2']}
#params = {'model__penalty': ['l2'], 'model__solver': ['lbfgs', 'liblinear']}
grid = GridSearchCV(logreg_pipeline, params, cv=cv, scoring='roc_auc')
grid.fit(train_x, train_y)
print(grid.best_score_)
#auc= 0.62
print(grid.best_params_)
#C=1, penalty=l2, lbfgs, defaults are best

#saving best model, after hp tuning
logreg_pipeline = Pipeline([("columntrans", tree_columntrans),
                            ("model", LogisticRegression(class_weight='balanced')),
                            ])
pickle.dump(logreg_pipeline, open('logreg_model3.pkl', 'wb'))

#------------EasyEnsemble hp tuning
easy_pipeline = Pipeline([("columntrans", tree_columntrans),
                          ("model", EasyEnsembleClassifier()),
                          ])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
params = {'model__sampling_strategy': ['auto', 0.5], 'model__n_estimators': [5, 10, 100]}
#params = {'model__n_estimators':[5,7,10]}
grid = GridSearchCV(easy_pipeline, params, cv=cv, scoring='roc_auc')
grid.fit(train_x, train_y)
print(grid.best_score_)
#auc= 0.62
print(grid.best_params_)
#n_estimators=5, sampling strategy=auto

#saving best model, after hp tuning
easy_pipeline = Pipeline([("columntrans", tree_columntrans),
                           ("model", EasyEnsembleClassifier(n_estimators=5)),
                          ])
pickle.dump(easy_pipeline, open('easyensemble_model3.pkl', 'wb'))

#-------------------------- testing the models on the test set
easy_pipeline = Pipeline([("columntrans", tree_columntrans),
                         ("model", EasyEnsembleClassifier(n_estimators=5)),
                         ])

easy_model = easy_pipeline.fit(train_x, train_y)
probabilities = easy_model.predict_proba(test_x)
predictions = easy_model.predict(test_x)
roc_auc_score(test_y, probabilities[:, 1])
recall_score(test_y, predictions)
precision_score(test_y, predictions)
#auc=0.60 (vs. 0.62 in CV, small overfit), recall= 0.51, precision=0.15


logreg_pipeline = Pipeline([("columntrans", reg_columntrans),
                         ("model", LogisticRegression(class_weight='balanced')),
                          ])
reg_model = logreg_pipeline.fit(train_x, train_y)
probabilities = reg_model.predict_proba(test_x)
predictions = reg_model.predict(test_x)
roc_auc_score(test_y, probabilities[:, 1])
recall_score(test_y, predictions)
precision_score(test_y, predictions)
#auc=0.60 (vs. 0.62 in CV, small overfit), recall=0.51, precision=0.14


# training and testing the main model on the test set for age group 70+
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
    ('clmn_passer', 'passthrough', [dmed_ix, medsdown_ix]),
],
)

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
#auc=0.61, recall=0.48, precision=0.15
#the general model has similar results, slightly lower recall