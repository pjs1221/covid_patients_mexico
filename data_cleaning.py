# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:16:54 2020

@author: shinp
"""

import pandas as pd
import datetime

df = pd.read_csv('Data/covid.csv')

#Delete Trivial/Incomplete Columns
del df['intubed']
del df['pregnancy']
del df['contact_other_covid']
del df['icu']

#Convert precondition values to numeric
pd.to_numeric(df.sex)
pd.to_numeric(df.patient_type)
pd.to_numeric(df.pneumonia)
pd.to_numeric(df.diabetes)
pd.to_numeric(df.copd)
pd.to_numeric(df.asthma)
pd.to_numeric(df.inmsupr)
pd.to_numeric(df.hypertension)
pd.to_numeric(df.cardiovascular)
pd.to_numeric(df.obesity)
pd.to_numeric(df.renal_chronic)
pd.to_numeric(df.tobacco)
pd.to_numeric(df.covid_res)

#Delete Unknown For Preconditions(represented by 97,98,99)
for i in range(97,100):
    df = df[df.sex != i]
    df = df[df.patient_type != i]
    df = df[df.pneumonia != i]
    df = df[df.diabetes != i]
    df = df[df.copd != i]
    df = df[df.asthma != i]
    df = df[df.inmsupr != i]
    df = df[df.hypertension != i]
    df = df[df.other_disease != i]
    df = df[df.cardiovascular != i]
    df = df[df.obesity != i]
    df = df[df.renal_chronic != i]
    df = df[df.tobacco != i]
    df = df[df.covid_res != i]


#Add Column for Days from First Symptoms to Days to Entry
df['entry_date'] = df['entry_date'].apply(lambda x: datetime.datetime.strptime(x,"%d-%m-%Y"))
df['date_symptoms'] = df['date_symptoms'].apply(lambda x: datetime.datetime.strptime(x,"%d-%m-%Y"))
df['days_to_entry'] = (df.entry_date-df.date_symptoms)
df['days_to_entry'] = df['days_to_entry'].apply(lambda x: x.days)

df['deceased'] = df['date_died'].apply(lambda x: 0 if x == '9999-99-99' else 1)


#Add Column for Days from First Symptoms to Date Died
df['date_died'] = df['date_died'].apply(lambda x: None if x == '9999-99-99' else datetime.datetime.strptime(x,"%d-%m-%Y"))
df['symptoms_to_death'] = (df.date_died-df.date_symptoms)
df['symptoms_to_death'] = df['symptoms_to_death'].apply(lambda x: x.days)


#Converting 2(false) to 0
df['sex'] = df['sex'].apply(lambda x: 0 if x == 2 else 1)
df['patient_type'] = df['patient_type'].apply(lambda x: 0 if x == 2 else 1)
df['pneumonia'] = df['pneumonia'].apply(lambda x: 0 if x == 2 else 1)
df['diabetes'] = df['diabetes'].apply(lambda x: 0 if x == 2 else 1)
df['copd'] = df['copd'].apply(lambda x: 0 if x == 2 else 1)
df['asthma'] = df['asthma'].apply(lambda x: 0 if x == 2 else 1)
df['inmsupr'] = df['inmsupr'].apply(lambda x: 0 if x == 2 else 1)
df['hypertension'] = df['hypertension'].apply(lambda x: 0 if x == 2 else 1)
df['other_disease'] = df['other_disease'].apply(lambda x: 0 if x == 2 else 1)
df['cardiovascular'] = df['cardiovascular'].apply(lambda x: 0 if x == 2 else 1)
df['obesity'] = df['obesity'].apply(lambda x: 0 if x == 2 else 1)
df['renal_chronic'] = df['renal_chronic'].apply(lambda x: 0 if x == 2 else 1)
df['tobacco'] = df['tobacco'].apply(lambda x: 0 if x == 2 else 1)
df['covid_res'] = df['covid_res'].apply(lambda x: 0 if x == 2 else 1)


df_out = df.drop(columns=[ 'entry_date','date_symptoms','date_died'])
df_out.to_csv('covid_data_cleaned.csv',index = False)
