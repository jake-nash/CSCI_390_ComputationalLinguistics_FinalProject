""" # FIXME intro, etc.
"""

from os import error
import numpy as np
import pandas as pd
from pandas.core.series import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



original_data = pd.read_csv("data/Original_dataset.csv")
useful_columns = original_data[["sourceLineText", "targetLineText", "sourceLineAbs", "targetLineAbs", "diffAbs_ins", "diffAbs_del", "ErrSet", "errorClang", "errorLLVM"]]

no_empty_cells = useful_columns.dropna(how="any")
no_empty_cells.to_csv("data/filtered_dataset.csv")

errorClang = pd.Series(no_empty_cells["errorClang"])
cleaned_errorClang = []
for message in errorClang:
    message = message.split()
    cleaned_message = []
    for j in range(len(message)):
        if message[j] in ["\n", "^", "^~","~", "~~", ":"]:
            continue
        else:
            cleaned_message.append(message[j])
    cleaned_message = " ".join(cleaned_message)
    cleaned_errorClang.append(cleaned_message)
cleaned_errorClang = pd.Series(cleaned_errorClang)

errorLLVM = pd.Series(no_empty_cells["errorLLVM"])
cleaned_errorLLVM = []
for message in errorLLVM:
    message = message.split()
    cleaned_message = []
    for j in range(len(message)):
        if message[j] in ["\n", "^", "^~","~", "~~", ":"]:
            continue
        else:
            cleaned_message.append(message[j])
    cleaned_message = " ".join(cleaned_message)
    cleaned_errorLLVM.append(cleaned_message)
cleaned_errorLLVM = pd.Series(cleaned_errorLLVM)



ErrSet = np.array(pd.Series(no_empty_cells["ErrSet"]))
ErrSet = pd.Series(ErrSet)

error_set_array = []
for i in ErrSet:
    err_sets = i.split(";")
    for j in err_sets:
            if j == '':
                    err_sets.remove(j)
    error_set_array.append(err_sets)

counter = 0
for i in range(3):
    for arr in error_set_array:
        if arr == ['']:
            error_set_array.remove(arr)
            counter += 1

ErrSet = pd.Series(error_set_array)



cleaned_dataset = pd.DataFrame({'ErrSet': ErrSet, 'errorClang': cleaned_errorClang, 'errorLLVM': cleaned_errorLLVM})
cleaned_dataset = cleaned_dataset.dropna(how="any")

err_array_copy = error_set_array.copy()
err_array_copy.sort()
num_error_sets = range(1, int(err_array_copy[-1][0]))
# new_columns = pd.DataFrame()
# index = 0
for i in num_error_sets:
    name = "Err_set_" + str(i)
#     cleaned_dataset = cleaned_dataset.assign(name)
#     # col = pd.Series(name=name, data=np.zeros)
#     # pd.concat(new_columns, col)
    cleaned_dataset.insert(loc=len(cleaned_dataset.columns), column=name, value=0)

# pd.concat(cleaned_dataset, new_columns)
cleaned_dataset = cleaned_dataset.copy()
cleaned_dataset.columns.sort_values()

cleaned_dataset.to_csv("data/cleaned_filtered_dataset.csv")

index = 0
for arr in cleaned_dataset["ErrSet"]:
    if index < 10:
        print("arr:", arr)
    for i in arr:
        i = i.strip()
        if index < 10:
            print("i:", i)
        column = "Err_set_" + str(i)
        cleaned_dataset.loc[index, column] = 1
    index += 1
    


cleaned_dataset.to_csv("data/cleaned_filtered_dataset.csv")




errsClang = []
for err in cleaned_dataset["errorClang"]:
    errsClang.append(err)
errsLLVM =[]
for err in cleaned_dataset["errorLLVM"]:
    errsLLVM.append(err)

tfidf_vect = TfidfVectorizer(ngram_range=(1,5))

tfidf_Clang = tfidf_vect.fit(errsClang)
tfidf_LLVM = tfidf_vect.fit(errsLLVM)



