import streamlit as st
import pandas as pd
# import sklearn
from sklearn.neighbors import NearestNeighbors
import re

# Chemin d'acces dans windows
# title_principals_movies = pd.read_csv(r"C:\Users\Lulu\Documents\Git\workaround\Projet_2\CSV\title_principals_movies.csv", sep=';')
# Chemin d'acces sous Linux (WSl terminal) /mnt/c
# title_principals_movies = pd.read_csv(r"/mnt/c/Users/Lulu/Documents/Git/workaround/Projet_2/CSV/title_principals_movies.csv", sep=';')
url = r'https://drive.google.com/file/d/1zOtYD44BXTOuAS96Bsc-zs_E8zVUD1J2/view?usp=sharing'
url = r'https://drive.google.com/uc?id=' + url.split('/')[-2]
title_principals_movies = pd.read_csv(url, sep=';')
title_principals_movies_copy = title_principals_movies

# st.set_page_config(layout='wide')


# Chargez l'image de fond
background_image_path =("Streamlite/contraste.jpg")  # Remplacez par le chemin de votre image
st.image(background_image_path, use_column_width=True)



url = 'https://image.tmdb.org/t/p/original'

# link = r"C:\Users\Lulu\Documents\Git\workaround\Projet_2\CSV\df_ml.csv"
# link = r"/mnt/c/Users/Lulu/Documents/Git/workaround/Projet_2/CSV/df_ml.csv"
link = r'https://drive.google.com/file/d/1RASIjC89JFVj4ucf_026Pcybm1nzI9Wn/view?usp=sharing'
link = r'https://drive.google.com/uc?id=' + link.split('/')[-2]
df = pd.read_csv(link)
df_copy = df.copy()

st.title('NewFlix')
st.divider()
col1, col2, col3 = st.columns(3)
with col2:
    st.image(r'https://drive.google.com/file/d/1N54BNQsuAQjO0YzbZynd6HUTWTjk4UNW/view?usp=sharing')



with st.sidebar:
    st.header("Fais ta selection")
    
    time = st.radio('Années',["All","2020's","2010's","2000's","90's","80's","70's","60's","50's","40's","30's"])

    if re.match(r'^\d{4}', time):
        df = df.loc[df['startYear'].astype(str).str.startswith(str(time[:3]))]
    elif re.match(r'^\d{2}', time):
        df = df.loc[df['startYear'].astype(str).str.startswith('19'+str(time[:1]))]
    st.write("")

    with st.form("filters1"):
        people = st.selectbox("Choisis un nom d'acteur ", title_principals_movies['primaryName'].unique(),index=None)
        #submitted11 = st.form_submit_button("Valider choix")
        submitted12 = st.form_submit_button("Valider")
    st.write("")
    if people:
        df = title_principals_movies.loc[title_principals_movies['primaryName'] == people]
        
    with st.form("filters"):
        movie = st.selectbox("Choisis un film", df["primaryTitle"].unique(),index=None)
        submitted1 = st.form_submit_button("Trouve moi un film")
        submitted2 = st.form_submit_button("Trouve moi ce titre ")
        

if submitted1:
    df = df_copy.copy()
    #X = df.drop(columns=['tconst','primaryTitle','poster_path','averageRating'])
    X = df.drop(columns=['tconst','primaryTitle','poster_path','averageRating','W_Note'])
    X_scaler = X
    # model = sklearn.neighbors(n_neighbors=4, algorithm='brute').fit(X)
    model = NearestNeighbors(n_neighbors=4, algorithm='brute').fit(X)
    X_index = df.loc[df['primaryTitle'].str.contains(movie)].index
    a = model.kneighbors(X_scaler.loc[X_index], return_distance=False)
    
    #col1.header(df.iloc[a[0][1]]['primaryTitle'])
    col1.subheader (df.iloc[a[0][1]]['startYear'])
    col1.image(url + df.iloc[a[0][1]]['poster_path'],use_column_width='auto')

    #col2.header(df.iloc[a[0][2]]['primaryTitle'])
    col2.subheader (df.iloc[a[0][2]]['startYear'])
    col2.image(url + df.iloc[a[0][2]]['poster_path'],use_column_width='auto')

    #col3.header(df.iloc[a[0][3]]['primaryTitle'])
    col3.subheader (df.iloc[a[0][3]]['startYear'])
    col3.image(url + df.iloc[a[0][3]]['poster_path'],use_column_width='auto')

if submitted2:
    df = df_copy.copy()
    df1 = df.loc[df['primaryTitle'].str.contains(movie, case = False)]
    cols = st.columns(4)
    for x in range (len(df1)):
        with cols[x % 4]:
            #st.subheader(df1.iloc[x,:]['primaryTitle'])
            st.subheader (df1.iloc[x,:]['startYear'])
            st.image(url + df1.iloc[x,:]['poster_path'],use_column_width='auto')

if submitted12:
    df = title_principals_movies.loc[title_principals_movies['primaryName'] == people]
    cols = st.columns(4)
    for x in range (len(df)):
        with cols[x % 4]:
            #st.subheader(df1.iloc[x,:]['primaryTitle'])
            st.subheader (df.iloc[x,:]['startYear'])
            st.text(df.iloc[x,:]['category'])
            st.image(url + df.iloc[x,:]['poster_path'],use_column_width='auto')

    #df.rename(columns={'category': 'job', 'primaryTitle': 'film','startYear' : 'année', 'averageRating' : 'note'}, inplace=True)
    #st.dataframe(df[['job','film','année','note']], hide_index=True)
    