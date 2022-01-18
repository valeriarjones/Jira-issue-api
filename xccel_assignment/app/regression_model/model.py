import datetime
import numpy as np
import pandas as pd
from sklearn import feature_selection, metrics, model_selection, preprocessing, tree
import yaml


def preprocess_data(df_day_counts: pd.DataFrame,
                    df_transitions: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """
    Function to process the data. It transforms date columns to datetime format and replaces nan values where possible.

    Args:
        df_day_counts: dataframe containing the daily counts for number of issues in all the possible states
                       ("In Progress", "Open", "Patch Available", "Reopened", "Resolved").
        df_transactions: dataframe containing JIRA-issues transitions and relative information.
    returns:
        processed dataframe
    """

    # To datetime format
    df_transitions["when"] = pd.to_datetime(df_transitions["when"])
    df_transitions["updated"] = pd.to_datetime(df_transitions["updated"])
    df_transitions["created"] = pd.to_datetime(df_transitions["created"])
    df_day_counts["day"] = pd.to_datetime(df_day_counts["day"])
    df_day_counts["date"] =df_day_counts["day"].dt.date

    # Process df_day_counts to merge to df_transactions
    status = df_day_counts["status"].unique()
    # One hot encode the statuses
    df_status = pd.get_dummies(df_day_counts["status"])
    # Multiply by their count to obtain a status_day_count dataframe
    df_status = df_status[status].multiply(df_day_counts["count"], axis=0)
    df_status.columns = ["{}_dcount".format(stat) for stat in status]
    # Concatenate the date column to then use to merge with df_transactions
    df_status = pd.concat([df_day_counts["date"], df_status], axis=1)

    # Groupby date to obtain a more compact df
    df_status = df_status.groupby("date").sum()
    df_status = df_status.reset_index()

    # from_status is NaN when transition is "Non-existent to Open", so I can safely replace NaNs with
    # "Non-existent" string
    df_transitions["from_status"].replace(np.nan, "Non-existent", inplace=True)

    # created is the only datetime column of df_transitions that is contained in date column of df_day_counts
    df_transitions["created_date"] = df_transitions["created"].dt.date
    df_transitions = df_transitions.merge(df_status, left_on="created_date", right_on="date")

    # Define numeric types and categorical/numerical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cat_col = ["to_status", "from_status", "resolution", "issue_type", "priority"]
    num_col = df_transitions.select_dtypes(include=numerics).columns

    # Replace NaN with reasonable values
    df_transitions[num_col] = df_transitions[num_col].replace(np.nan, 0)
    df_transitions["resolution"].replace(np.nan, "No resolution", inplace=True)

    return df_transitions, cat_col, num_col

def split_data(df_transitions: pd.DataFrame) -> pd.DataFrame:
    """
    """
    # Select only issues that have transitioned to resolved at least once
    keys = df_transitions.loc[df_transitions["transition"].str.contains("Resolved"),"key"].unique()

    # Filter on those issues and ignore those entries that have transitioned from resolved to another status
    df_select = df_transitions[(df_transitions["key"].isin(keys)) & (df_transitions["from_status"]!="Resolved")]

    # Calculate resolve date for every issue (in the reduced dataframe)
    df_max = df_select[["key", "when"]].groupby("key").max().rename(columns={"when": "resolve_date"})
    # Merge to reduced dataframe
    df_select = df_select.merge(df_max, right_on="key", left_on="key")

    return df_select, keys

def define_target(df: pd.DataFrame) -> pd.Series:
    """
    Function to get target column from data. The target column will timedelta between date issue was resolved
    and when the issue changed to to_status, transformed to fraction of a day.
    Args:
        df: transitions dataframe containing only transitions up to issue goes to_status resolved.
    returns:
        target column
    """
    # Timedelta between issue transitioned to resolved
    df["timedelta"] = df["resolve_date"] - df["when"]

    # Timedelta to float as fraction of a day
    df["timedelta"] = df["timedelta"].dt.total_seconds() / datetime.timedelta(days=1).total_seconds()

    return df["timedelta"]

def feature_engineering(df: pd.DataFrame,
                        cat_col: list,
                        num_col: list,
                        predict: bool=False, minmax_scaler: object=None) -> pd.DataFrame:
    """
    """
    # ohe categorical columns
    df_ohe = pd.get_dummies(df[cat_col])

    # Scale numerical columns
    num_values = df[num_col].values

    # If feature engineering for train-test and not final predict, fit minmax scaler
    if not predict:
        minmax_scaler = preprocessing.MinMaxScaler()
        minmax_scaler = minmax_scaler.fit(num_values)

    # Scale numerical values
    num_values_scaled = minmax_scaler.transform(num_values)
    df_scaled = pd.DataFrame(num_values_scaled)
    df_scaled.columns=num_col
    df_scaled.set_index(df.index, inplace=True)

    # Concatenate with ohe data
    df_scaled = pd.concat([df_ohe, df_scaled], axis=1)

    return df_scaled, minmax_scaler

def select_features(df: pd.DataFrame,
                    target: pd.Series,
                    dec_tree: object,
                    step: int=5,
                    min_feat: int=7,
                    cv: int=5) -> list:
    """
    """
    # Select best features with recursive feature elimination
    rfe = feature_selection.RFECV(dec_tree, step=step, min_features_to_select=min_feat, cv=cv)
    rfe = rfe.fit(df, target)
    sel_features = df.columns[rfe.support_].to_list()

    return sel_features


def run():

    # Load configuration
    with open("../xccel_assignment/data/config.yaml", "r") as config:
        config = yaml.safe_load(config)

    verbose = config["verbose"]

    # Read csv to pandas dataframe
    df_issues = pd.read_csv("{}{}".format(config["issues"]["path"], config["issues"]["name"]),
                            delimiter=config["issues"]["delim"])
    df_day_counts = pd.read_csv("{}{}".format(config["day_counts"]["path"], config["day_counts"]["name"]),
                                delimiter=config["day_counts"]["delim"])
    df_transitions = pd.read_csv("{}{}".format(config["transitions"]["path"], config["transitions"]["name"]),
                                 delimiter=config["transitions"]["delim"])

    # Define target column name
    target_col = "timedelta"

    # Process data to prepare for feature engineering
    df_transitions, cat_col, num_col = preprocess_data(df_day_counts, df_transitions)

    # Get only entries of issues that get resolved and history until resolution
    df_select, keys = split_data(df_transitions)

    # Get target column
    df_select[target_col] = define_target(df_select)

    # Drop entries with to_status Resolved
    df_select = df_select.loc[df_select["to_status"]!="Resolved"]

    # Ohe categorical columns and scale numerical columns
    df_processed, scaler = feature_engineering(df_select, cat_col, num_col)

    # Train-test split
    df_train, df_test, target_train, target_test = model_selection.train_test_split(df_processed,
                                                                                    df_select[target_col],
                                                                                    test_size=0.2)
    # Define instance of DecisionTreeRegressor
    dec_tree = tree.DecisionTreeRegressor()

    if verbose:
        print("Performing feature selection...")
    # Perform feature selection (in this case RFECV)
    sel_features = select_features(df_train, target_train, dec_tree)

    if verbose:
        print("Done! Selected features are: {}".format(sel_features))

    # Fit model with selected features
    dec_tree = dec_tree.fit(df_train[sel_features], target_train)

    # Predict on test data
    predictions = dec_tree.predict(df_test[sel_features])

    # Evaluation metrics
    evs = metrics.explained_variance_score(target_test, predictions)
    mae = metrics.mean_absolute_error(target_test, predictions)
    mse = metrics.mean_squared_error(target_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(target_test, predictions))

    if verbose:
        print("Explained Variance Score: {}".format(evs))
        print('Mean Absolute Error: {}'.format(mae))
        print('Mean Squared Error: {}'.format(mse))
        print('Root Mean Squared Error: {}'.format(rmse))

    # Get entries that have not been used to train-test
    df_t = df_transitions[(~df_transitions["key"].isin(keys)) |
                          ~(df_transitions["from_status"]!="Resolved")].copy(deep=True)

    if verbose:
        print("Predicting on issues not used to train-test...")
    # Transform
    df_proces_test, scaler = feature_engineering(df_t, cat_col, num_col, predict=True,
                                                        minmax_scaler=scaler)
    # Predict
    target_pred = dec_tree.predict(df_proces_test[sel_features])

    # Transform predictions to timdelta
    target_pred = pd.DataFrame(target_pred * datetime.timedelta(days=1).total_seconds()) \
           .apply(lambda x: datetime.timedelta(seconds=x[0]), axis=1)
    target_pred.index=df_t.index

    # Add timedelta to when date to get resolution date prediction
    df_t["resolve_date"] = target_pred + df_t["when"]
    df_final = df_issues.join(df_t.groupby("key")["resolve_date"].max(), on="key")
    df_final.loc[df_final["status"] == "Resolved", "resolve_date"] = df_final.loc[
                                                                     df_final["status"] == "Resolved", "updated"]
    # Select only relevant columns
    df_final = df_final[["key", "status", "resolve_date"]]
    df_final.columns = ["issue", "current_status", "prediction_resolution_date"]
    df_final["prediction_resolution_date"] = pd.to_datetime(df_final["prediction_resolution_date"], utc=True)

    if verbose:
        print("Done!")

    return df_final

if __name__ == "__main__":
    run()