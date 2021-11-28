import matplotlib.pyplot as plt
import numpy as np
from numpy import isnan
import pandas as pd
from pathlib import Path

# import re
# import seaborn as sns

# from datetime import datetime
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn import preprocessing
# from sklearn.decomposition import PCA

ROOT_PATH = Path().absolute()
DATASET_PATH = ROOT_PATH / 'data' / 'XY_train.csv'
##------------read the files---------------------##
# data_xy_train = pd.read_csv(DATASET_PATH)
data_xy_train = pd.read_csv(r'C:\Users\Owner\OneDrive\Documents\ml\XY_train.csv')


# data_x_test = pd.read_csv(r'C:\Users\shirg\Downloads\X_test.csv')
# data_json = pd.read_json(r'C:\Users\shirg\Desktop\לימוד מכונה\partOne\Jason_File.json')


# ##------------EDA---------------##
# ##--CITY--##
# plt.hist(data_xy_train['city'], bins=2000, color='darkblue', )
# plt.title("City Histogram", fontsize=20)
# plt.xlabel('city', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()
#
# #
# ##--city_development_index--##
# plt.hist(data_xy_train['city_development_index'], bins=1000, color='darkblue')
# plt.title("city_development_index Histogram", fontsize=20)
# plt.xlabel('city_development_index', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()
# #
# ##--gender--##

def clean_nan(s: pd.Series) -> pd.Series:
    return s[~s.isna()]


def plot_hist(s: pd.Series, title: str, x_label: str, y_label: str = 'Frequency', bins=50, color='darkblue'):
    plt.hist(s, bins=bins, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.show()


def plot_scatter(s: pd.Series, yi: pd.Series, title1: str, title2: str, ):
    plt.scatter(x=s, y=y)
    plt.title("Interaction between $title1 and $title2", fontsize=20)
    plt.xlabel('title1')
    plt.ylabel('title2')
    plt.show()

#
# if __name__ == '__main__':
#
#     gender = clean_nan(s=data_xy_train['gender'])
#     plot_hist(s=gender,
#               title='Gender histogram',
#               x_label='gender')
#     plot_hist(s=data_xy_train['training_hours'],
#               title="training_hours Histogram",
#               x_label='training_hours')
#
#
#
# city=clean_nan(s=data_xy_train['city'])
# plot_hist(s=city,
#           title='city histogram',
#           x_label='city')
# cityD=clean_nan(s=data_xy_train['city_development_index'])
# plot_hist(s=cityD,
#           title='city development index histogram',
#           x_label='cityD')
# relevntExp=clean_nan(s=data_xy_train['relevent_experience'])
# plot_hist(s=relevntExp,
#           title='relevent experience histogram',
#           x_label='relevent experience')

# university=clean_nan(s=data_xy_train['enrolled_university'])
# plot_hist(s=university,
#           title='enrolled university histogram',
#           x_label='enrolled university ')
#
# eduacation=clean_nan(s=data_xy_train['education_level'])
# plot_hist(s=eduacation,
#           title='education level histogram',
#           x_label='education level')
#


# diceplin=clean_nan(s=data_xy_train['major_discipline'])
# plot_hist(s=diceplin,
#           title='major discipline histogram',
#           x_label='major discipline')
# exp=clean_nan(s=data_xy_train['experience'])
# plot_hist(s=exp,
#           title='experience histogram',
#           x_label='experience')

# companyS=clean_nan(s=data_xy_train['company_size'])
# plot_hist(s=companyS,
#           title='company size histogram',
#           x_label='company size')

# companyT=clean_nan(s=data_xy_train['company_type'])
# plot_hist(s=companyT,
#           title='company type histogram',
#           x_label='company type')

# last_new_job=clean_nan(s=data_xy_train['last_new_job'])
# plot_hist(s=last_new_job,
#           title='last_new_job histogram',
#           x_label='last_new_job')
#
#
# training_hours=clean_nan(s=data_xy_train['training_hours'])
# plot_hist(s=training_hours,
#           title='training hours histogram',
#           x_label='training hours')

x=data_xy_train['training_hours']
y=data_xy_train['last_new_job']
plot_scatter(s=x, yi=y, title1='training_hours', title2='last_new_job')





# ##-------Pre Prossecing--------##
# ##--Missing Values--##
# #-empty Categories-#
# data_xy_train['cat_fixed'] = data_xy_train.apply(
#     lambda row: data_xy_train[data_xy_train['description'] == row['description']].iloc[0]['categoryId'] if np.isnan(row['categoryId']) else row['categoryId'],
#     axis=1)
# data_xy_train['categoryId'] = data_xy_train['cat_fixed']
#
# #-empty likes-#
# df_cat_id = data_xy_train[data_xy_train['categoryId'] == 10].copy()
# mean_cat_id10 = df_cat_id['likes'].mean()
# data_xy_train['likes'] = data_xy_train['likes'].fillna(mean_cat_id10)
#
# #-empty descriptions-#
# data_xy_train['description_fixed'] = data_xy_train.apply(
#     lambda row: row['description'] if row['description'] == '' else row['title'], axis=1)
# data_xy_train['description'] = data_xy_train['description_fixed']
# data_xy_train.drop('description_fixed', axis='columns', inplace=True)
# data_xy_train.drop('cat_fixed', axis='columns', inplace=True)
#
# #-empty tags-#
# x = data_xy_train.isnull()
# for key in data_xy_train.keys():
#     for i in range(len(data_xy_train[key])):
#         if x[key][i] or data_xy_train[key][i] == "[None]":
#             if key == "categoryId":
#                 if data_xy_train["title"][i] == data_xy_train["title"][i - 1] and data_xy_train["publishedAt"][i] == \
#                         data_xy_train["publishedAt"][i - 1] and not x[key][i - 1]:
#                     data_xy_train[key][i] = data_xy_train[key][i - 1]
#             if key == "tags":
#                 categoryId = data_xy_train["categoryId"][i]
#                 for item in data_json["items"]:
#                     if int(item["id"]) == int(categoryId):
#                         data_xy_train[key][i] = item["snippet"]["title"]
