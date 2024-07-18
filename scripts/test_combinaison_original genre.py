import pandas as pd
import numpy as np
import itertools

from colorama import init, Fore, Back, Style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

link_main =r"C:\Users\Lulu\Documents\Git\Projet_2\df_clean\full_df.csv"
link_opinion = r"C:\Users\Lulu\Documents\Git\Projet_2\df_clean\df_genres.csv"


def getCombinationList(column_list: list) -> list[list]:
    print(f'{Fore.YELLOW}{Style.BRIGHT}[ PROCESSING ] >> Generating combination list')
    print(f'{Fore.CYAN}{Style.BRIGHT}[ INFO ] >> Dataframe column list --> {Fore.BLUE}{column_list}')
    final_list = []

    for n in range(len(column_list) + 1):
        for subset in itertools.combinations(column_list, n):
            final_list.append(list(subset))

    return final_list

def getCombinedDf(combination: list, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print(f'{Fore.YELLOW}{Style.BRIGHT}[ PROCESSING ] >> Generating dataframe for combination --> {Fore.MAGENTA}{combination}')
    # combined_df = df.drop(columns=combination)
    combined_df = df.loc[:, combination]
    return combined_df

def calculateDistance(df1: pd.core.frame.DataFrame, df2: pd.core.frame.DataFrame) -> list[float]:
    X_train, X_test, y_train, y_test = train_test_split(df1, df2, train_size = 0.75, shuffle = True, random_state=42)
    modelRL2 = LinearRegression().fit(X_train, y_train)

    trainScore = modelRL2.score(X_train, y_train)
    testScore = modelRL2.score(X_test, y_test)
    distance = trainScore - testScore

    return distance, trainScore, testScore


def main():
    init(autoreset=True)
    df_main = pd.read_csv(link_main, sep='\t')
    df_opinion = pd.read_csv(link_opinion, sep='\t')

    df_reco = pd.merge(df_main,
                df_opinion,
                how="left",
                left_on='tconst',
                right_on='tconst')
    
    df_clean = df_reco.drop(columns= ['tconst', 'primaryTitle', 'originalTitle', 'backdrop_path', 'original_language', 'poster_path',
                           'production_countries', 'release_date', 'status', 'spoken_languages', 'video',
                            'production_companies_name', 'production_companies_country']).dropna()
    df_clean=pd.concat([df_clean,pd.get_dummies(df_clean[['genres']])],axis=1)

    list_col_df = list(df_clean.columns)
    list_col_df.remove('vote_average')

    final_list = getCombinationList(list_col_df)

    final_result = []

    for combination in final_list[1:]:
        combinedDf = getCombinedDf(combination, df_clean)
        dist, train, test = calculateDistance(combinedDf, df_clean['vote_average'])

        final_result.append({"combination": combination, "distance": dist, "trainScore": train, "testScore": test, "addition": test+train})
        
    sorted_result = sorted(final_result, key=lambda d: d['addition'], reverse=True)
    print(sorted_result[:5])
    
if __name__ == "__main__":
    main()










