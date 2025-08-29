import pandas as pd

#{"company_name", "review_date", "text", "stars", "category", "label"

def get_df(filepath, category):
    df = pd.read_csv(filepath)
    df = df[["title", "stars", "text", "publishedAtDate"]]
    df["category"] = category
    df = df.rename(columns={"publishedAtDate": "review_date", "title": "company_name"})
    df["text"].fillna('', inplace=True)
    return df

def display(df):
    print(df.head())
    print(df.columns)

def save_csv(df, filepath):
    df.to_csv(filepath, index=False)  

educationDF = get_df("datasets/education_reviews.csv", "education")
fnbDF = get_df("datasets/fnb_reviews.csv", "food and beverages")
healthcareDF = get_df("datasets/healthcare_reviews.csv", "healthcare")
retailDF = get_df("datasets/retail_reviews.csv", "retail")

display(educationDF)
display(fnbDF)
display(healthcareDF)
display(retailDF)

save_csv(educationDF, "modified/education_reviews.csv")
save_csv(fnbDF, "modified/fnb_reviews.csv")
save_csv(healthcareDF, "modified/healthcare_reviews.csv")
save_csv(retailDF, "modified/retail_reviews.csv")
