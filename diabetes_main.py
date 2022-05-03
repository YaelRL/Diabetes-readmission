import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

# ------1. frame the problem and look at the big picture
# ------2. get the data, take a quick look at the data structure and set aside a test set
# df=pd.read_csv('train.csv')
df = pd.read_csv('data/diabetic_data.csv')
df.replace('?', np.nan, inplace=True)
df.replace('NULL', np.nan, inplace=True)

#deleting observations for patients who expired/discharged to hospice
died=[11,13,14,19,20,21]
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

df = filter_rows_by_values(df, "discharge_disposition_id", died)

#keeping only one visit per patient
df = df.drop_duplicates("patient_nbr")
#69,990 unique visits

df.info()
# features weight, payer_code, medical specialty have more than 40% missing values
df.drop(["weight", "payer_code", "medical_specialty"],axis=1, inplace=True)
df.describe()

#len(df["patient_nbr"].unique())

#unifying small categories
df["race"].value_counts()
mask = np.logical_or(df["race"] == "Hispanic", df["race"] == "Asian")
df.loc[mask, "race"]="Other"

#formatting categorical variables, from character to numeric
df["gender"].value_counts()
df["gender"].replace("Male", 0, inplace=True)
df["gender"].replace("Female", 1, inplace=True)
df["gender"].replace("Unknown/Invalid", 0, inplace=True)

#unifying small categories
df["age"].value_counts()
mask = np.logical_or(df["age"] == "[10-20)", df["age"] == "[0-10)"]
df.loc[mask, "age"] = "[0-30)"
df.loc[df["age"] == "[20-30)", "age"] = "[0-30)"


df["admission_type_id"].value_counts()
#there are 10 entries with admission type 4=newborn
#checking consistency
mask=df["admission_type_id"]==4
newborn=df.loc[mask]
newborn["age"].head
#age doesnt match, better convert to na
#converting admission types 4,5,6 and 8 to np.nan
mask1 = np.logical_or(df["admission_type_id"] == 4, df["admission_type_id"] == 5)
mask2 = np.logical_or(df["admission_type_id"] == 6, df["admission_type_id"] == 8)
mask= np.logical_or(mask1 , mask2)
df.loc[mask, "admission_type_id"] = np.nan
df.loc[df["admission_type_id"] == 7, "admission_type_id"] = 1
#10% missing values
#1=Emergency
#2=Urgent
#3=Elective

df["discharge_disposition_id"].value_counts()
#checking consistency
mask=df["discharge_disposition_id"]==10
neonate= df.loc[mask]
neonate["age"]
#age doesnt match, converting to np.nan
df.loc[df["discharge_disposition_id"] == 10, "discharge_disposition_id"] = np.nan
#unifying small categories
df.loc[df["discharge_disposition_id"] ==12, "discharge_disposition_id"] = 1
df.loc[df["discharge_disposition_id"] == 27, "discharge_disposition_id"] = 2
df.loc[df["discharge_disposition_id"] == 16, "discharge_disposition_id"] = 4
df.loc[df["discharge_disposition_id"] == 17, "discharge_disposition_id"] = 4
df.loc[df["discharge_disposition_id"] == 9, "discharge_disposition_id"] = 2
df.loc[df["discharge_disposition_id"] == 24, "discharge_disposition_id"] = 3
df.loc[df["discharge_disposition_id"] == 15, "discharge_disposition_id"] = 2
df.loc[df["discharge_disposition_id"] == 8, "discharge_disposition_id"] = 1
df.loc[df["discharge_disposition_id"] == 28, "discharge_disposition_id"] = 2
df.loc[df["discharge_disposition_id"] == 23, "discharge_disposition_id"] = 3
df.loc[df["discharge_disposition_id"] == 7, "discharge_disposition_id"] = 2
df.loc[df["discharge_disposition_id"] == 4, "discharge_disposition_id"] = 3
df.loc[df["discharge_disposition_id"] == 25, "discharge_disposition_id"] = np.nan
df.loc[df["discharge_disposition_id"] == 5, "discharge_disposition_id"] = np.nan
df.loc[df["discharge_disposition_id"] == 22, "discharge_disposition_id"] = 2
df.loc[df["discharge_disposition_id"] == 8, "discharge_disposition_id"] = 1
df.loc[df["discharge_disposition_id"] == 18, "discharge_disposition_id"] = np.nan
df.loc[df["discharge_disposition_id"] == 6, "discharge_disposition_id"] = 1
#1=home
#2=short term hospitalization
#3=long-term care facility


df["admission_source_id"].value_counts()

# 'admission_source_id',
# 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
# 'num_medications', 'number_outpatient', 'number_emergency',
# 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
# 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
# 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
# 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
# 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
# 'insulin', 'glyburide-metformin', 'glipizide-metformin',
# 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
# 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted'

df["readmitted"].value_counts()
# 55% not readmitted, 35% reamitted >30, 11% readmitted <30

df["time_in_hospital"].hist(bins=50, figsize=(20, 15))

plt.show()
#long right tail
