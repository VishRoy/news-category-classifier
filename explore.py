import pandas as pd

# df = pd.read_csv('sample_data.csv')
df_news = pd.read_json('data/News_Category_Dataset_v3.json', lines=True)
# print(df.shape)
# print(df.columns.tolist())
# print(df.head())
# print(df["category"].value_counts())


# print(df.isnull().sum())
# print(df.duplicated().sum())
print(df_news.shape)
print(df_news.isnull().sum())
print(df_news.columns.tolist())
print(df_news["category"].value_counts())
print(df_news.duplicated().sum())
