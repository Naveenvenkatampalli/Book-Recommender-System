import numpy as np
import pandas as pd
df=pd.read_csv('Books.csv',error_bad_lines=False,encoding='latin-1')
df1=pd.read_csv('Ratings.csv',error_bad_lines=False,encoding='latin-1')
df2=pd.read_csv('Users.csv',error_bad_lines=False,encoding='latin-1')
df=df[['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher']]
x=df1['User-ID'].value_counts()>200
y=x[x].index
y
df1=df1[df1['User-ID'].isin(y)]
ratings_books=df1.merge(df,on='ISBN')
ratings_books.head()
num_ratings=ratings_books.groupby('Book-Title')['Book-Rating'].count().reset_index()
num_ratings.rename(columns={'Book-Rating':'num_of_ratings'},inplace=True)
final=ratings_books.merge(num_ratings,on='Book-Title')
final=final[final['num_of_ratings']>=50]
final.drop_duplicates(['User-ID','Book-Title'],inplace=True)
book_pivot=final.pivot_table(columns='User-ID',index='Book-Title',values='num_of_ratings')
book_pivot.fillna(0,inplace=True)
from scipy.sparse import csr_matrix
book_sparse=csr_matrix(book_pivot)
from sklearn.neighbors import NearestNeighbors
model=NearestNeighbors(algorithm='brute')
model.fit(book_sparse)
def recommend_book(book_name):
    book_id=np.where(book_pivot.index==book_name)[0][0]
    distances,suggestions=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
    for i in range(len(suggestions)):
            print(book_pivot.index[suggestions[i]])

recommend_book('A Case of Need')
