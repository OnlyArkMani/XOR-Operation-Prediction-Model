import pandas as pd
df = pd.read_csv(r"C:\Projects\Assignment\data-100000-100-4-rnd.csv")
print(df.iloc[0, :10])   # print first 10 values
print(df.dtypes.head(10))
