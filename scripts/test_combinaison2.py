import pandas as pd
import numpy as np
import itertools
import multiprocessing as mp

from timeit import default_timer as timer
from datetime import timedelta
from colorama import init, Fore, Back, Style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

link_main = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/weather_main_2018.csv"
link_opinion = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/weather_opinion_2018.csv"

def getCombinationList(columnList: list) -> list[list]:
    print(f'{Fore.YELLOW}{Style.BRIGHT}[ PROCESSING ] >> Generating combination list')
    print(f'{Fore.CYAN}{Style.BRIGHT}[ INFO ] >> Dataframe column list --> {Fore.BLUE}{columnList}')
    final_list = []

    for n in range(len(columnList) + 1):
        for subset in itertools.combinations(columnList, n):
            final_list.append(list(subset))
    return final_list


def getRestrictedDf(combination: list, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print(f'{Fore.YELLOW}{Style.BRIGHT}[ PROCESSING ] >> Generating dataframe for combination --> {Fore.MAGENTA}{combination}')
    return df.loc[:, combination]


def calculateCorr(df1: pd.core.frame.DataFrame, df2: pd.core.frame.DataFrame) -> list[float]:
    X_train, X_test, y_train, y_test = train_test_split(df1, df2, train_size = 0.75, shuffle = True, random_state=42)
    modelRL2 = LinearRegression().fit(X_train, y_train)
    trainScore = modelRL2.score(X_train, y_train)
    testScore = modelRL2.score(X_test, y_test)
    distance = trainScore - testScore
    return distance, trainScore, testScore

def process_combination(combination, df, keyColumn):
    df_comb = getRestrictedDf(combination, df)
    dist, train, test = calculateCorr(df_comb, df[keyColumn])
    return {"combination": combination, "distance": dist, "trainScore": train, "testScore": test, "addition": test + train}

def generateResults(combinationList: list, df: pd.core.frame.DataFrame, keyColumn: str) -> list:
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(process_combination, [(comb, df, keyColumn) for comb in combinationList])
    return results


def sortResults(results: list, rankingParameter: str, reverse: bool=True) -> list:
    return sorted(results, key=lambda d: d[rankingParameter], reverse=reverse)


def displayResults(sortedResults: list, range: int):
    count = 1
    for result in sortedResults[:range]:
        print(f"{Fore.GREEN}{Style.BRIGHT}Number {Fore.RED}{count}{Fore.GREEN} result is the combination : {Fore.MAGENTA}{result.get('combination')}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Total Test/Train score is {Fore.CYAN}{result.get('addition')}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Train score : {Fore.CYAN}{result.get('trainScore')}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Test score : {Fore.CYAN}{result.get('testScore')}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Distance score : {Fore.CYAN}{result.get('distance')}")
        print("")
        count = count + 1

def main():
    init(autoreset=True)
    start = timer()
    df_main = pd.read_csv(link_main)
    df_opinion = pd.read_csv(link_opinion)

    df2018_full = pd.merge(df_main,
                df_opinion,
                how="left",
                left_on='DATE',
                right_on='date')
    
    df_clean = df2018_full.drop(columns= ['DATE', 'OPINION', 'date']).dropna()

    list_col_df = list(df_clean.columns)
    list_col_df.remove('SUNHOUR')

    combinationList = getCombinationList(list_col_df)
    results = generateResults(combinationList[1:], df_clean, "SUNHOUR")
    sorted_results = sortResults(results, 'addition')

    displayResults(sorted_results, 3)

    end = timer()
    print(f'{Fore.GREEN}{Style.BRIGHT}Program total duration : {Fore.BLUE}{str(timedelta(seconds=end-start))}')
    
if __name__ == "__main__":
    main()



