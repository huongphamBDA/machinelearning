import datetime
import os
import sys
import warnings

import numpy
import pandas
import process_baseball_pham
import utils_pham
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# CREATE OUTPUT FOLDER
plots_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
)
print(plots_dir)
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)


def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "r", linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate (FPR)", fontsize=16)
    plt.ylabel("True Positive Rate (TPR)", fontsize=16)


def main():
    # Increase pandas print viewport (so we see more on the screen)
    pandas.set_option("display.max_rows", 10)
    pandas.set_option("display.max_columns", 500)
    pandas.set_option("display.width", 1_000)

    data_set_name = "baseball"

    # 1. INPUT DATA & PROCESS DATA
    utils_pham.print_heading("1. Input and Process Data")
    data_df, X, y = process_baseball_pham.process_baseball()

    # Drop these features based on the feature inspection and correlation done in part 3 and 4 below
    # . drop the first batch of bad features
    drop_cols_1 = [
        "BA_1diff",
        "BA_2diff",
        "BA_3diff",
        "Slug_1diff",
        "Slug_2diff",
        "Slug_3diff",
        "Field_Error_diff",
        "Field_Error_diff2",
        "plateApperance_diff",
    ]  # len 9
    X.drop(drop_cols_1, inplace=True, axis=1)
    data_df.drop(drop_cols_1, inplace=True, axis=1)

    # . drop the second batch of bad features
    drop_cols_2 = [
        "stadium_Sun Life Stadium",
        "stadium_Jacobs Field",
        "venue_Angel Stadium of Anaheim",
        "league_interleague at AL",
        "stadium_Busch Stadium",
        "venue_Busch Stadium",
        "stadium_Target Field",
        "venue_Target Field",
        "league_interleague at NL",
        "stadium_Safeco Field",
        "venue_Safeco Field",
        "weather_sunny",
        "time_am",
        "time_pm",
        "winddir_Out to LF",
        "stadium_Wrigley Field",
        "venue_Wrigley Field",
        "stadium_Rangers Ballpark in Arlington",
        "stadium_Citizens Bank Park",
        "venue_Citizens Bank Park",
        "stadium_Tropicana Field",
        "venue_Tropicana Field",
        "stadium_PNC Park",
        "venue_PNC Park",
        "winddir_In from LF",
        "winddir_Varies",
        "venue_Progressive Field",
        "stadium_Turner Field",
        "venue_Turner Field",
        "stadium_Angel Stadium of Anaheim",
        "league_AL",
        "stadium_Rogers Centre",
        "venue_Rogers Centre",
        "stadium_Chase Field",
        "venue_Chase Field",
        "league_NL",
        "stadium_Citi Field",
        "venue_Citi Field",
        "weather_cloudy",
    ]  # len 39
    X.drop(drop_cols_2, inplace=True, axis=1)
    data_df.drop(drop_cols_2, inplace=True, axis=1)

    # . drop the third batch of bad features - due to high correlation b/w cont xi -cont xj
    drop_cols_3 = [
        "BA_2away",
        "BA_3ratio",
        "BA_2home",
        "BA_1away",
        "Slug_1ratio",
        "plateApperance_diff2",
        "BA_2ratio",
        "Slug_2ratio",
        "BA_2ratio",
        "BA_1home",
        "BA_3away",
    ]
    print(len(drop_cols_3))
    X.drop(drop_cols_3, inplace=True, axis=1)
    data_df.drop(drop_cols_3, inplace=True, axis=1)

    # . drop the forth batch of bad features - due to high correlation b/w stadium and venue cat-cat features
    drop_cols_4 = [col for col in X.columns if "venue" in col]
    print(len(drop_cols_4))  # 30
    X.drop(drop_cols_4, inplace=True, axis=1)
    data_df.drop(drop_cols_4, inplace=True, axis=1)

    # . drop "winddir_None" due to high correation b/w cat-cat
    X.drop("winddir_None", inplace=True, axis=1)
    data_df.drop("winddir_None", inplace=True, axis=1)

    # Some Brute Force effort: Please look at my report for explanation. It's decided in part 5 of this code
    # . compute new feature slug_hit = Slug_3ratio/hits_ratio, then drop Slug_3ratio

    # X["slug_hit"] = X["Slug_3ratio"] + X["hits_ratio"]
    # data_df["slug_hit"] = data_df["Slug_3ratio"] + data_df["hits_ratio"]
    # X.drop("Slug_3ratio", inplace=True, axis=1)
    # data_df.drop("Slug_3ratio", inplace=True, axis=1)
    # X["slug_hit"] = X["slug_hit"].astype(float)
    # data_df["slug_hit"] = data_df["slug_hit"].astype(float)

    # . compute new feature plateApperance = plateApperance_ratio2/ plateApperance_ratio, then drop plateApperance_ratio
    # X["plateApperance"] = X["plateApperance_ratio2"] / X["plateApperance_ratio"]
    # data_df["plateApperance"] = data_df["plateApperance_ratio2"] / data_df["plateApperance_ratio"]
    # X.drop("plateApperance_ratio", inplace=True, axis=1)
    # data_df.drop("plateApperance_ratio", inplace=True, axis=1)
    #
    # # . compute new feature plate_BA = plateApperance_ratio2/BA_3home, then drop BA_3home
    # X["plate_BA"] = X["plateApperance_ratio2"] / X["BA_3home"]
    # data_df["plate_BA"] = data_df["plateApperance_ratio2"] / data_df["BA_3home"]
    # X.drop("BA_3home", inplace=True, axis=1)
    # data_df.drop("BA_3home", inplace=True, axis=1)
    #
    # X = X.dropna()
    # data_df = data_df.dropna()

    predictors = X.columns.values.tolist()
    print("Number of predictors: ", len(predictors))

    # 2. DETERMINE COLUMN TYPES
    utils_pham.print_heading("2. Determine column types after encoding")
    categorical_df, continuous_df = utils_pham.split_data_set(X, predictors)
    cat_list = categorical_df.columns.tolist()
    cont_list = continuous_df.columns.tolist()

    # 3. FEATURE INSPECTION (x vs. y)
    utils_pham.print_heading("3. Feature Inspection")
    # The response is a boolean so we need three plot types: . Heatplot (for categorical/boolean
    # predictor). Violin/ Distribution plots on continuous predictor grouped by response
    population_mean = y.mean()

    f_path = []
    p_val = []
    t_val = []
    html_link_list = []
    msd_ranking = []
    w_msd_ranking = []
    plot_lk = []
    col_names = [
        "Response",
        "Predictor",
        "Predictor_is_Cat_or_Cont",
        "Plot_Predictor_vs_Response",
        "t_value",
        "p_value",
        "Visualize_t_value_and_p_value",
        "Random_Forest_Variable_Importance",
        "Difference_with_Mean_of_Response",
        "Weighted_Difference_with_Mean_of_Response",
        "Plot_the_Difference_with_Mean_of_Response",
    ]
    feature_inspection_output_df = pandas.DataFrame(columns=col_names)
    feature_inspection_output_df["Predictor"] = X.columns

    for col in predictors:
        tval, pval, html_link = utils_pham.cal_plot_pval_tscore(data_df, col, plots_dir)
        t_val.append(tval)
        p_val.append(pval)
        html_link_list.append(html_link)
        MSD_ranking, w_MSD_ranking, html_file = utils_pham.cal_plot_dmr_uw_w(
            data_df, col, population_mean, plots_dir
        )
        msd_ranking.append(MSD_ranking)
        w_msd_ranking.append(w_MSD_ranking)
        plot_lk.append(f'<a href="{html_file}" target="_blank"> {html_file} </a>')
        indices_list = list(
            numpy.where(feature_inspection_output_df["Predictor"] == col)[0]
        )

        if col in cont_list:
            feature_inspection_output_df.loc[
                indices_list[0], "Predictor_is_Cat_or_Cont"
            ] = "Continuous"
            filename = f"Distribution_plot_Cont_predictor_({col})_vs_Cat_response.html"
            utils_pham.create_distribution_plot(data_df, col, plots_dir, filename)
            importance_list = utils_pham.get_feature_importance(
                plots_dir, continuous_df, y
            ).tolist()
            feature_inspection_output_df.loc[
                indices_list[0], "Random_Forest_Variable_Importance"
            ] = importance_list[indices_list[0]]
        elif col in cat_list:
            feature_inspection_output_df.loc[
                indices_list[0], "Predictor_is_Cat_or_Cont"
            ] = "Categorical"
            filename = f"Heatmap_Cat_predictor_({col})_vs_Cat_response.html"
            utils_pham.create_heatmap(data_df, col, filename, plots_dir)
            feature_inspection_output_df.loc[
                indices_list[0], "Random_Forest_Variable_Importance"
            ] = "N/A"

        f_path.append(f'<a href="{filename}" target="_blank"> {filename} </a>')

    feature_inspection_output_df.loc[:, "Response"] = "HomeTeamWins"
    feature_inspection_output_df["Plot_Predictor_vs_Response"] = f_path
    feature_inspection_output_df["t_value"] = t_val
    feature_inspection_output_df["p_value"] = p_val
    feature_inspection_output_df["Visualize_t_value_and_p_value"] = html_link_list
    feature_inspection_output_df["Difference_with_Mean_of_Response"] = msd_ranking
    feature_inspection_output_df[
        "Weighted_Difference_with_Mean_of_Response"
    ] = w_msd_ranking
    feature_inspection_output_df["Plot_the_Difference_with_Mean_of_Response"] = plot_lk

    feature_inspection_output_df.sort_values(
        by=["Weighted_Difference_with_Mean_of_Response", "t_value"],
        ascending=False,
        inplace=True,
    )
    output_html = feature_inspection_output_df.to_html(escape=False, index=False)

    # write html to file
    text_file = open(f"{plots_dir}/feature_inspection_output.html", "w")
    text_file.write(output_html)
    text_file.close()
    print(feature_inspection_output_df.head())

    # 4. xi vs. xj CORRELATION
    utils_pham.print_heading("4. xi vs. xj CORRELATION")
    cont_cont_correlation_file = (
        utils_pham.create_table_and_matrix_cont_cont_correlation(
            plots_dir, continuous_df
        )
    )
    cont_cat_correlation_file = utils_pham.create_table_and_matrix_cont_cat_correlation(
        plots_dir, categorical_df, continuous_df
    )

    (
        cat_cat_correlation_stchuprow,
        cat_cat_correlation_cramer,
    ) = utils_pham.create_table_and_matrix_cat_cat_correlation(
        plots_dir, categorical_df
    )

    utils_pham.create_html_table_with_links(
        dataset_name=data_set_name,
        cont_cont_file=cont_cont_correlation_file,
        cont_cat_file=cont_cat_correlation_file,
        cat_cat_stchuprow_file=cat_cat_correlation_stchuprow,
        cat_cat_cramer_file=cat_cat_correlation_cramer,
        plots_dir=plots_dir,
    )

    # Look for the name of bad features
    # . any features that have "diff" in their names, except streak_diff cause it has high RF var importance
    # batch_cols_1 = [col for col in X.columns if 'diff' in col]

    # . any features that have Weighted_Difference_with_Mean_of_Response too small
    small_w_dmr_df = feature_inspection_output_df.loc[
        feature_inspection_output_df["Weighted_Difference_with_Mean_of_Response"]
        < 4.489885e-06
    ]
    batch_cols_2 = small_w_dmr_df["Predictor"].tolist()
    print(batch_cols_2)
    print(len(batch_cols_2))

    # I decided what features to remove after doing Feature Inspection and Feature Correlation (part 3 and 4), but I
    # came back to remove them in part 1 instead of doing it here to save time running code.

    # 5. BRUTE FORCE (xi, xj vs. y)
    utils_pham.print_heading("5. BRUTE FORCE (xi, xj vs. y)")
    bin_num = 10

    # cont-cont predictors
    utils_pham.create_msd_table_cont_cont(
        bin_num, continuous_df, data_df, population_mean, "HomeTeamWins", plots_dir
    )

    # cont-cat predictors
    utils_pham.create_msd_table_cont_cat(
        bin_num,
        categorical_df,
        continuous_df,
        data_df,
        population_mean,
        "HomeTeamWins",
        plots_dir,
    )

    # cat-cat predictors
    utils_pham.create_msd_table_cat_cat(
        bin_num, categorical_df, data_df, population_mean, "HomeTeamWins", plots_dir
    )

    # . compute new feature slug_hit = Slug_3ratio/hits_ratio, then drop Slug_3ratio
    # . compute new feature plateApperance = plateApperance_ratio2/ plateApperance_ratio, then drop plateApperance_ratio
    # . compute new feature plat_BA = plateApperance_ratio2/BA_3home, then drop BA_3home

    # 6. BUILD MODELS
    utils_pham.print_heading("6. BUILD MODELS")

    # . Train/Test Split
    split_date = datetime.datetime(2009, 1, 1)
    df_training = data_df.loc[data_df["local_date"] <= split_date]
    df_test = data_df.loc[data_df["local_date"] > split_date]

    y_train = df_training["HomeTeamWins"]
    X_train = df_training.loc[
        :, (df_test.columns != "HomeTeamWins") & (df_test.columns != "local_date")
    ]
    y_test = df_test["HomeTeamWins"]
    X_test = df_test.loc[
        :, (df_test.columns != "HomeTeamWins") & (df_test.columns != "local_date")
    ]

    # 6.1. Logistic Regression
    clf_log = LogisticRegression().fit(X_train, y_train)
    y_pred1 = clf_log.predict(X_test)

    # 6.2. Decision Tree Model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred2 = clf.predict(X_test)

    # 6.3. KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred3 = knn.predict(X_test)

    # 6.4. Random Forest Model
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred4 = rf.predict(X_test)

    # 7. EVALUATE MODELS
    utils_pham.print_heading("7. EVALUATE MODELS")

    # 7.1. Accuracy score
    accuracy1 = metrics.accuracy_score(y_test, y_pred1)
    accuracy2 = metrics.accuracy_score(y_test, y_pred2)
    accuracy3 = metrics.accuracy_score(y_test, y_pred3)
    accuracy4 = metrics.accuracy_score(y_test, y_pred4)

    # 7.2. Confusion Matrix
    cm1 = metrics.confusion_matrix(y_test, y_pred1)
    cm2 = metrics.confusion_matrix(y_test, y_pred2)
    cm3 = metrics.confusion_matrix(y_test, y_pred3)
    cm4 = metrics.confusion_matrix(y_test, y_pred4)

    # 7.3. Precision Score
    precision1 = metrics.precision_score(y_test, y_pred1)
    precision2 = metrics.precision_score(y_test, y_pred2)
    precision3 = metrics.precision_score(y_test, y_pred3)
    precision4 = metrics.precision_score(y_test, y_pred4)

    # 7.4. ROC/ ROC_AUC Scores
    roc_score1 = metrics.roc_auc_score(y_test, y_pred1)
    roc_score2 = metrics.roc_auc_score(y_test, y_pred2)
    roc_score3 = metrics.roc_auc_score(y_test, y_pred3)
    roc_score4 = metrics.roc_auc_score(y_test, y_pred4)
    false_positive_rate1, true_positive_rate1, thresholds1 = metrics.roc_curve(
        y_test, y_pred1
    )
    false_positive_rate2, true_positive_rate2, thresholds2 = metrics.roc_curve(
        y_test, y_pred2
    )

    plt.figure(figsize=(14, 7))
    plot_roc_curve(false_positive_rate1, true_positive_rate1)
    plt.title("ROC Curve - Logistic Regression")
    plt.show()

    plt.figure(figsize=(14, 7))
    plot_roc_curve(false_positive_rate2, true_positive_rate2)
    plt.title("ROC Curve - Decision Tree")
    plt.show()

    # 7.5. Output all the results in one table
    results = pandas.DataFrame(
        {
            "Model": ["Logistic Regression", "Decision Tree", "KNN", "Random Forest"],
            "Accuracy": [accuracy1, accuracy2, accuracy3, accuracy4],
            "Confusion Matrix": [cm1, cm2, cm3, cm4],
            "Precision": [precision1, precision2, precision3, precision4],
            "ROC_AUC": [roc_score1, roc_score2, roc_score3, roc_score4],
        }
    )
    result_df = results.sort_values(by=["Accuracy", "ROC_AUC"], ascending=False)
    results_html = result_df.to_html(escape=False, index=False)
    results_file = open(f"{plots_dir}/model_evaluation_results.html", "w")
    results_file.write(results_html)
    results_file.close()
    print(result_df)


if __name__ == "__main__":
    sys.exit(main())
