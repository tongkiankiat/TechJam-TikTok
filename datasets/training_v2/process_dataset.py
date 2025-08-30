import pandas as pd

def get_df(filepath):
    df = pd.read_csv(filepath)
    return df

def display(df):
    print(df.head())
    print(df.columns)

def save_csv(df, filepath):
    df.to_csv(filepath, index=False)  

df = get_df("datasets/training_v2/new_data/withtfidf")
save_csv(df, "datasets/training_v2/new_data/withtfidf.csv")
display(df)
# df1["label"] = 1
# df_old = get_df("datasets/training_v2/merged_reviews_fixed.csv")
# df_old = df_old[df_old["label"] != 1]
# combined = pd.concat([df_old, df1])
# combined_no_reviewername = combined.drop("reviewer_name", axis = 1)
# display(combined_no_reviewername)
# print(combined_no_reviewername.shape)
# print(combined_no_reviewername["label"].unique())