import pandas as pd
import numpy as np
import itertools

#itertools = librarie qui permet d'utiliser des fonctions d'iteration de maniere efficiente et plus 
# perfectionnee qu'une fonction built in.

from colorama import init, Fore, Back, Style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

link_main = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/weather_main_2018.csv"
link_opinion = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/weather_opinion_2018.csv"

# fonction qui permet de creer ma liste de combinaison
def getCombinationList(columnList) :
    print(f'{Fore.YELLOW}{Style.BRIGHT}[ PROCESSING ] >> Generating combination list')
    print(f'{Fore.CYAN}{Style.BRIGHT}[ INFO ] >> Dataframe column list --> {Fore.BLUE}{columnList}')
    combinationList = []
    # j'itere sur toutes les colonnes de ma liste de colonnes pour creer une nouvelle liste qui contient toutes les combinaisons possible
    for n in range(len(columnList) + 1):
        # jutilise la lib itertools pour generer ces conmbinaisons
        for subset in itertools.combinations(columnList, n):
            # chaque combinaison est ajoutee a ma liste finale (la combinaison est egalement sous forme de liste)
            combinationList.append(list(subset))
    # je retourne ma liste de combinaison
    return combinationList

# fonction qui permet de calculer mes differents scores
def calculateCorr(dfX, dfY):
    X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, train_size = 0.75, shuffle = True, random_state=42)
    modelRL2 = LinearRegression().fit(X_train, y_train)

    trainScore = modelRL2.score(X_train, y_train)
    testScore = modelRL2.score(X_test, y_test)
    distance = trainScore - testScore

    return distance, trainScore, testScore


def main():
    init(autoreset=True) # remet a la couleur standard automatiquement(partie coloration)
    df_main = pd.read_csv(link_main)
    df_opinion = pd.read_csv(link_opinion)

    df2018_full = pd.merge(df_main,
                df_opinion,
                how="left",
                left_on='DATE',
                right_on='date')
    
    df_clean = df2018_full.drop(columns= ['DATE', 'OPINION', 'date']).dropna()

    list_col_df = list(df_clean.columns) # Je recupere la liste des colonnes de mon DF sous forme de liste
    list_col_df.remove('SUNHOUR') # je retire 'SUNHOUR' de ma liste de colonnes. La liste de colonnes va servir d'input pour creer ma liste de combinaison

    combinationList = getCombinationList(list_col_df) # j'appelle la fonction qui permet de generer ma liste de combinaison (en fonction de ma liste de colonnes)

    results = [] # j'initialise une liste vide pour stocker les resultats de correlation plus tard

    # j'itere sur ma liste de combinaison
    for combination in combinationList[1:]:
        # je cree un DF qui correspond a ma combinaison courante
        print(f'{Fore.YELLOW}{Style.BRIGHT}[ PROCESSING ] >> Generating dataframe for combination --> {Fore.MAGENTA}{combination}')
        selectedDf = df_clean.loc[:, combination]
        # je calcule mon trainScorem testScore, et ma distance
        dist, train, test = calculateCorr(selectedDf, df_clean['SUNHOUR'])
        # j'ajoute chaque resultat sous forme de dict dans ma liste creee precedemment
        results.append({"combination": combination, "distance": dist, "trainScore": train, "testScore": test, "addition": test+train})
    # je classe mes resultats en fonction du parametre 'addition' (trainScore + testScore) par ordre decroissant
    sorted_result = sorted(results, key=lambda d: d['addition'], reverse=True)
    # j'affiche les 5 meilleurs resultats
    print(sorted_result[:5])

    
if __name__ == "__main__":
    main()










