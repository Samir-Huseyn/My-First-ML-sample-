import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("my_file (1).csv", sep=None, engine="python")
print(df.info())
print(df.shape)
print(df.describe())
df.columns = ["rank", "peak", "all time peak", "actual gross", "adjusted gross", "artist", "tour title","year", "shows", "average gross", "ref"]
print(df.columns.to_list())
df["actual gross"] = df["actual gross"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("[b]", "", regex=False).str.replace("[e]", "", regex=False).astype(float)
df["average gross"] = df["average gross"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
df["adjusted gross"] = df["adjusted gross"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
print(df[['artist', 'adjusted gross', 'actual gross', 'average gross']].head(10))
df.drop(columns=["ref"], inplace= True)
print(df.isnull().sum())
df["all time peak"] = df["all time peak"].str.split("[").str[0]
df["peak"] = df["peak"].str.split("[").str[0]
df['peak'] = pd.to_numeric(df['peak'], errors='coerce')
df["all time peak"] = pd.to_numeric(df["all time peak"], errors="coerce")
df["peak"] = df["peak"].fillna(0)
df["all time peak"] = df["all time peak"].fillna(0)
df['tour title'] = df['tour title'].str.split('†').str[0].str.split('‡').str[0].str.split("*").str[0].str.strip()
df['tour title'] = df['tour title'].str.split('[').str[0].str.strip()
df["year"] = df["year"].str.split("-").str[0].str.strip()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
median_year = df["year"].median()
df["year"] = df["year"].fillna(median_year).astype(int)
df["shows"] = pd.to_numeric(df["shows"], errors="coerce")
print(df.info())
print(df[["artist", "tour title","actual gross", "shows", "average gross"]].head(10))
best_per_show = df.sort_values(by="average gross", ascending=False)
print(best_per_show)
artist_performance = df.groupby("artist").agg({
    "actual gross" : "sum",
    "shows" : "sum",
    "average gross" : "mean"

}).sort_values(by="actual gross", ascending=False)
print(f"total performance: {artist_performance}")

df.to_csv("cleaned_concert_data.csv", index=False)


X = df[['shows']] 
y = df['actual gross']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"datas for learning: {len(X_train)}")
print(f"Datas for checking: {len(X_test)}")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model Learned")
new_travel = pd.DataFrame({"shows" : [100] })
guess = model.predict(new_travel)
print(f"gross per guess 100 shows: {guess}")

plt.figure()
df.groupby("artist")["actual gross"].sum().sort_values(ascending=False).plot(kind="bar", color="skyblue")
plt.title("Gross for artist")
plt.xlabel("artist")
plt.ylabel("actual gross")
plt.show()