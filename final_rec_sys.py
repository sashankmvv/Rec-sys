# %%
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# %%
def recommendation(csv_file): #valuation,location,sq_feet etc.
    df=pd.read_csv(csv_file+".csv",index_col='Users')
    df=df.fillna(0)
    return df

# %%

# %%
def convert_int(df):
    for i in df.columns:
        df[i]=df[i].astype('int')
    return df

# %%
def calculate_percentages(df):
    total=0
    for i in df.columns:
        total=df[i]+total
    df = df.pipe(lambda x: (x*100).div(total, axis='index'))
    df=df.fillna(0)
    return df

# %%
def standardize(row):
    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row

def correlation(df_std,df):

    sparse_df = sparse.csr_matrix(df_std.values)
    corrMatrix = pd.DataFrame(cosine_similarity(sparse_df),index=df.columns,columns=df.columns)
    return corrMatrix

# %%
def get_similar(warehouse,percentage):
    similar_score = corrMatrix[warehouse]*(percentage-50.5) #50..5 is the avg of percentage (0-100)
    similar_score = similar_score.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_score

# %%
def final_recommendation(portfolio):

    similar_scores = pd.DataFrame()
    for warehouse,percentages in portfolio:
        similar_scores = similar_scores.append(get_similar(warehouse,percentages))

    similar_scores.head(10)
    return similar_scores



# %%
rec=['valuation','location','square_feet']
portfolio_valuation = [("small_cap",2),("large_cap",98),("mid_cap",0)]
portfolio_location = [("Rural",2),("Semi_Urban",0),("Urban",98)]
portfolio_square_feet = [("Small_warehouse",2),("Medium_warehouse",98),("Large_warehouse",0)]
portfolio=[portfolio_valuation,portfolio_location,portfolio_square_feet]
appended_data = []

for i in range(len(rec)):
    df=recommendation(rec[i])
    df=convert_int(df)
    df=calculate_percentages(df)
    df_std = df.apply(standardize).T
    corrMatrix=correlation(df_std,df)
    # similar_score=get_similar("bluechip",2)
    
    # for i in range(len(portfolio:
    similar_scores=final_recommendation(portfolio[i])
    appended_data.append(pd.Series(list(similar_scores.sum().sort_values(ascending=False)[:2].index)))



appended_data = pd.concat(appended_data,axis=1).reset_index(drop=True)

    
appended_data.rename(columns={0: 'Valuation', 1: 'Location',2:'Square_feet'}, inplace=True)

print(appended_data)



# %%
import numpy as np

items=pd.read_csv("warehouses.csv")

Valuations=["large_cap","mid_cap","small_cap"]
square_f=["Small_warehouse","Medium_warehouse","Large_warehouse"]
loc=["Urban","Semi_Urban","Rural"]

items["Valuation"] = np.random.choice(Valuations, len(items))
items["Square_feet"] = np.random.choice(square_f, len(items))
items["Location"] = np.random.choice(loc, len(items))

items.to_csv("warehouses.csv",index=False)

print(items)

# %%
print(appended_data)
# %%
appended_data.merge(items,'inner')
# df2, df1 = df2.append(t).drop_duplicates(keep=False) , df1.append(t).drop_duplicates(keep=False)
# %%
