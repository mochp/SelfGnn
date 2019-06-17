# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd


# BIG5
path = "data/" + "BIG5" + "/" + "data.csv"
ans_csv = pd.read_csv(path, sep="\t")
feature = ans_csv.iloc[:, :7]
answers = ans_csv.iloc[:, 7:]


ages = list(feature["age"].copy())
for i, age in enumerate(ages):
    if age < 18:
        ages[i] = 1
    elif age < 28 and age >= 18:
        ages[i] = 2
    elif age < 40 and age >= 28:
        ages[i] = 3
    elif age < 66 and age >= 40:
        ages[i] = 4
    else:
        ages[i] = 5

feature["age"] = pd.Series(ages)

country = list(feature["country"].copy())
country_list = set(country)

dic = {}
dic['OO'] = 0
for cou in country_list:
    dic[cou] = country.count(cou)

dic['OO'] = dic.pop("(nu")
for cou in list(dic.keys()):
    if dic[cou] < 90:
        dic['OO'] += dic.pop(cou)

cou_dic = {}
for i, cd in enumerate(list(dic.keys())):
    cou_dic[cd] = i


for i, age in enumerate(country):
    try:
        country[i] = cou_dic[age]
    except BaseException:
        country[i] = cou_dic["OO"]

feature["country"] = pd.Series(country)


feature = feature.drop(19064)
answers = answers.drop(19064)
answers.to_excel("answers.xlsx", index=None)
feature.to_excel("feature.xlsx", index=None)
