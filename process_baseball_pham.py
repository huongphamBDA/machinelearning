import os

import numpy
import pandas
import sqlalchemy
import utils_pham
from sklearn.preprocessing import OneHotEncoder

# CREATE OUTPUT FOLDER
plots_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
)


def process_baseball():

    # 1. INPUT DATA FROM SQL TO PYTHON
    db_user = "root"
    db_pass = "finalproject"  # pragma: allowlist secret
    db_host = "mariadb"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    sql_engine = sqlalchemy.create_engine(connect_string)

    query = "SELECT * FROM finalProject_data_addedDate"
    data_df = pandas.read_sql_query(query, sql_engine)  # shape: 3073 * 47

    input_html = data_df.to_html()
    text_file = open(f"{plots_dir}/data_from_SQL.html", "w")
    text_file.write(input_html)
    text_file.close()

    # data_df = pandas.read_csv('./final_project_data/finalProject_data_addedDate_2021-12-18 16:45:50.txt', sep="|")

    # 2. PROCESS DATA & DETERMINE COLUMN TYPES
    utils_pham.print_subheading("Data Info")
    print(data_df.info)
    print("\nAll columns in the original dataset:\n", data_df.columns)
    utils_pham.print_subheading("Sum null values of original columns")
    print(data_df.isnull().sum())
    # streak_ratio has 1172 Null rows, need to drop this feature

    # Drop streak_ratio
    data_df.drop("streak_ratio", inplace=True, axis=1)

    # Count empty values of some features that I saw having empty values when I explored the data
    print("\nCount empty values")
    print((data_df["home_throwinghand"].values == "").sum())  # 6
    print((data_df["away_throwinghand"].values == "").sum())  # 5
    print((data_df["status"].values == "").sum())  # 0

    # Replace empty values with NaN
    data_df["home_throwinghand"].replace("", numpy.nan, inplace=True)
    data_df["away_throwinghand"].replace("", numpy.nan, inplace=True)
    data_df = data_df.dropna()

    # After removing empty values, the status column has only one value left "F" --> Remove status feature
    data_df.drop("status", inplace=True, axis=1)

    # Get temperature number and remove the 'degrees'
    data_df["tempF"] = data_df["temp"].str[:2].astype(float)
    data_df.drop("temp", inplace=True, axis=1)

    # Convert hits_diff to float
    data_df["hits_diff"] = pandas.to_numeric(data_df["hits_diff"])

    # Assume if games took place indoor, the wind was 0 mph; remove "mph" from the cell values; convert to float
    data_df.loc[data_df.wind == "Indoors", "wind"] = "0 mph"
    data_df["wind"] = data_df["wind"].str.replace(r" mph", "")
    data_df["wind"] = data_df["wind"].astype(float)

    # Rename some columns:
    data_df.rename(
        columns={"stadium_name": "stadium", "venue_short": "venue"}, inplace=True
    )

    # Determine column types
    y_original = data_df["HomeTeamWins"]
    X_original = data_df.loc[
        :,
        (data_df.columns != "HomeTeamWins")
        & (data_df.columns != "local_date")
        & (data_df.columns != "game_id")
        & (data_df.columns != "home_team_id")
        & (data_df.columns != "away_team_id")
        & (data_df.columns != "series_game_number"),
    ]
    predictors_original = X_original.columns.values.tolist()
    print("\nOriginal Predictors:", predictors_original)

    utils_pham.print_subheading(
        "Determine columns of original dataset (before encoding)"
    )
    cat_orig_df, cont_orig_df = utils_pham.split_data_set(
        X_original, predictors_original
    )
    print("\nOriginal Categorical_df:\n", cat_orig_df.head())
    print("\nOriginal Continuous_df:\n", cont_orig_df.head())
    cat_list = cat_orig_df.columns.tolist()
    # cont_list = cont_orig_df.columns.tolist()

    # Number of unique values in each categorical feature (ampm 2 unique values, league 4 unique values,
    # home/away_throwing_hand 3 (one is empty), winddir 10, overcast 9, stadium 37, venue 43)
    print(
        "\nNumbers of original categorical features' unique values:\n",
        cat_orig_df.nunique(axis=0),
    )
    # Look at unique values of categorical features
    for col in cat_list:
        print(
            f"\nUnique values in the original categorical column {col}: ",
            data_df[col].unique(),
        )

    # Split cat_orig_df into two dataframes, one for stadium and venue, one for the others
    cat_1_df = cat_orig_df[["stadium", "venue"]].copy()
    cat_2_df = cat_orig_df[
        [
            "ampm",
            "league",
            "home_throwinghand",
            "away_throwinghand",
            "winddir",
            "overcast",
        ]
    ].copy()

    # The 30 most frequent stadiums and venues
    print(
        "\nValue count of the 30 most frequent stadiums:\n",
        cat_1_df.stadium.value_counts().sort_values(ascending=False).head(30),
    )
    print(
        "\nValue count of the 30 most frequent venues:\n",
        cat_1_df.venue.value_counts().sort_values(ascending=False).head(30),
    )
    # Create lists of the top 30 categories
    top_30_stadium = [
        x
        for x in cat_1_df.stadium.value_counts()
        .sort_values(ascending=False)
        .head(30)
        .index
    ]
    top_30_venue = [
        x
        for x in cat_1_df.venue.value_counts()
        .sort_values(ascending=False)
        .head(30)
        .index
    ]
    # Convert 30 most frequent categories into one-hot-encoding, the rest will turn to zeros
    utils_pham.add_cols_for_encoded_top_categories(cat_1_df, "stadium", top_30_stadium)
    utils_pham.add_cols_for_encoded_top_categories(cat_1_df, "venue", top_30_venue)
    cat_1_df.drop(["stadium", "venue"], inplace=True, axis=1)

    # Encoding categorical features
    cat_2_array = cat_2_df.to_numpy()
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(cat_2_array)
    cat_csr_matrix = one_hot_encoder.transform(cat_2_array)
    cat_2_df_new = pandas.DataFrame(cat_csr_matrix.toarray())
    cat_2_df_new.columns = one_hot_encoder.get_feature_names(
        ["time", "league", "home_hand", "away_hand", "winddir", "weather"]
    )
    cat_2_df_new = cat_2_df_new.astype(int)

    # Dataset after being processed:
    X = pandas.concat([cont_orig_df, cat_1_df, cat_2_df_new], axis=1, join="inner")
    data_df_new = pandas.concat(
        [X, y_original, data_df["local_date"]], axis=1, join="inner"
    )
    y = data_df_new["HomeTeamWins"]
    utils_pham.print_subheading("All columns after encoding")
    print(data_df_new.columns.values.tolist())
    print(data_df_new.dtypes)

    # write html to file
    data_html = data_df_new.to_html()
    text_file = open(f"{plots_dir}/data_python.html", "w")
    text_file.write(data_html)
    text_file.close()

    return data_df_new, X, y


def main():
    print("\nThis is process_baseball module.")
    process_baseball()


if __name__ == "__main__":
    main()
