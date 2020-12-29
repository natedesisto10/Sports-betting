import numpy as np
import CleanData1
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def main():
    current_week = 17
    weeks = list(range(1,current_week + 1))
    year = 2020
    pred_games_df, comp_games_df = CleanData1.prep_test_train(current_week,weeks,year)
    msk = np.random.rand(len(comp_games_df)) < 0.8

    train_df = comp_games_df[msk]
    test_df = comp_games_df[~msk]
    
    X_train = train_df.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week','result'])
    y_train = train_df[['result']] 
    X_test = test_df.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week','result'])
    y_test = test_df[['result']]
    X_train, X_test, y_train, y_test, train_df, test_df, pred_games_df = random_forest(X_train, X_test, y_train, y_test, train_df, test_df, pred_games_df)
    X_train, X_test, y_train, y_test, train_df, test_df, pred_games_df = logreg(X_train, X_test, y_train, y_test, train_df, test_df, pred_games_df)
    
def random_forest(X_train, X_test, y_train, y_test, train_df, test_df, pred_games_df):
    clf = RandomForestClassifier(n_estimators=100,max_depth=5, min_samples_split=2,
                          min_samples_leaf=1, max_features='auto', bootstrap=True)

    clf.fit(X_train, np.ravel(y_train.values))
    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:,1]

    CleanData1.display(y_pred, test_df)
    print()
    print(accuracy_score(y_test,np.round(y_pred)))

    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:,1]
    print()
    CleanData1.display(y_pred, pred_games_df)
    print()
    CleanData1.display2(y_pred, pred_games_df)
    
def logreg(X_train, X_test, y_train, y_test, train_df, test_df, pred_games_df):
    clf = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True, 
                   intercept_scaling=1, class_weight='balanced', random_state=None, 
                   solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0)

    clf.fit(X_train, np.ravel(y_train.values))
    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:,]

    CleanData1.display(y_pred, test_df)
    print()
    print(accuracy_score(y_test,np.round(y_pred)))

    y_pred = clf.predict_proba(X_test)
    y_pred = y_pred[:,]
    print()
    CleanData1.display(y_pred, pred_games_df)
    print()
    CleanData1.display2(y_pred, pred_games_df)    
    
    
if __name__ == '__main__':
    main()

