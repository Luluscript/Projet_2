import pandas as pd
import itertools
import multiprocessing as mp
from alive_progress import alive_bar

from timeit import default_timer as timer
from datetime import timedelta
from colorama import init, Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

link_main = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/weather_main_2018.csv"
link_opinion = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/weather_opinion_2018.csv"


def getCombinationList(columnList: list) -> list[list]:
    print(f'{Fore.YELLOW}{Style.BRIGHT}[ PROCESSING ] >> Generating combination list')
    print(f'{Fore.CYAN}{Style.BRIGHT}[ INFO ] >> Dataframe column list --> {Fore.BLUE}{columnList}')
    chaining = itertools.chain.from_iterable(itertools.combinations(columnList, x) for x in range(len(columnList)+1))
    combinationList = [list(subset) for _ , subset in enumerate(chaining)]
    return combinationList


def calculateCorr(dfX: pd.DataFrame, dfY: pd.Series, trainSize: float=0.75, isShuffle: bool=True, randomState: int=42) -> tuple[float]:
    X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, train_size=trainSize, shuffle=isShuffle, random_state=randomState)
    modelRL2 = LinearRegression().fit(X_train, y_train)
    trainScore = modelRL2.score(X_train, y_train)
    testScore = modelRL2.score(X_test, y_test)
    distance = trainScore - testScore
    return distance, trainScore, testScore


def processCombination(*args):
    if len(args) == 1:
        args = args[0]
    combination, df, keyColumn = args
    df_restricted = df.loc[:, combination]
    dist, train, test = calculateCorr(df_restricted, df[keyColumn])
    return {"combination": combination, "distance": dist, "trainScore": train, "testScore": test, "addition": test + train}


def generateResults(combinationList: list, df: pd.DataFrame, keyColumn: str, bar=None) -> list:
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        args = [(combination, df, keyColumn) for combination in combinationList]
        if bar:
            for result in pool.imap_unordered(processCombination, args):
                results.append(result)
                bar()
        else:
            results = pool.starmap(processCombination, args)
    return results


def sortResults(results: list, rankingParameter: str, reverse: bool=True) -> list:
    return sorted(results, key=lambda d: d[rankingParameter], reverse=reverse)


def displayResults(sortedResults: list, range: int):
    for index, result in enumerate(sortedResults[:range], 1):
        print("")
        print(f"{Fore.GREEN}{Style.BRIGHT}Number {Fore.RED}{index}{Fore.GREEN} result is the combination : {Fore.MAGENTA}{result.get('combination')}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Total Test/Train score is {Fore.CYAN}{result.get('addition')}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Train score : {Fore.CYAN}{result.get('trainScore')}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Test score : {Fore.CYAN}{result.get('testScore')}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}Distance score : {Fore.CYAN}{result.get('distance')}")


def main():
    init(autoreset=True)
    fast=False

    remove_columns = ['DATE', 'OPINION', 'date']
    point = "SUNHOUR"

    start = timer()
    df_main = pd.read_csv(link_main)
    df_opinion = pd.read_csv(link_opinion)

    df_merged = pd.merge(df_main,
                df_opinion,
                how="left",
                left_on='DATE',
                right_on='date')
    
    df_clean = df_merged.drop(columns = remove_columns).dropna()

    list_col_df = list(df_clean.columns)
    list_col_df.remove(point)

    combinationList = getCombinationList(list_col_df)

    if fast:
        results = generateResults(combinationList[1:], df_clean, point)
    else:  
        with alive_bar(len(combinationList), force_tty=True, title="Processing Combinations", length=40) as bar:
            results = generateResults(combinationList[1:], df_clean, point, bar)

    displayResults(sortResults(results, 'addition'), 3)

    end = timer()
    print(f'\n{Fore.GREEN}{Style.BRIGHT}Program total duration : {Fore.BLUE}{str(timedelta(seconds=end-start))}')
    
if __name__ == "__main__":
    main()