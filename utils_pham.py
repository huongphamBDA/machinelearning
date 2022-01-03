import os
import warnings

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# CREATE OUTPUT FOLDER
plots_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
)


def print_heading(title):
    print("\n")
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


def print_subheading(title):
    print("\n")
    print("-" * 50)
    print(title)
    print("-" * 50)
    return


def fill_na(data):
    if isinstance(data, pandas.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


#################################
# Step 1: DETERMINE COLUMN TYPES
#################################
def _determine_column_type_object(df, col):
    """
    Case of data frame column is type 'object'

    :param df: Dataframe
    :param col: Current column being examined
    :return: A string of 'boolean'
    """
    bool_array_yn = numpy.array([1, 0])
    bool_array_ny = numpy.array([0, 1])
    unique_values = df[col].unique()
    total_unique_values = df[col].value_counts().count()

    if total_unique_values == 2 and (
        numpy.array_equal(unique_values, bool_array_yn)
        or (numpy.array_equal(unique_values, bool_array_ny))
    ):
        print(f"{col} is boolean")
        return "boolean"

    print(f"{col} is categorical")
    return "categorical"


def _determine_column_type_category(df, col):
    """
    Case of data frame column type is 'category'.
    There's a chance that it's boolean.

    :param df: Dataframe
    :param col: Current column being examined
    :return: Either "boolean" or "categorical" value
    """
    unique_values = df[col].unique()
    total_unique_values = df[col].value_counts().count()

    if total_unique_values == 2:
        # underline values are string type e.g, 'YES', 'NO', or 'MALE', 'FEMALE', etc.
        if isinstance(unique_values[0], str):
            # compare yes-no or no-yes
            if [unique_values[0].lower(), unique_values[1].lower()] == [
                "yes",
                "no",
            ] or [unique_values[0].lower(), unique_values[1].lower()] == ["no", "yes"]:
                print(f"{col} is boolean")
                return "boolean"
            else:
                print(f"{col} is categorical")
                return "categorical"
        elif isinstance(unique_values[0], int):
            if [unique_values[0], unique_values[1]] == [0, 1] or [
                unique_values[0],
                unique_values[1],
            ] == [1, 0]:
                print(f"{col} is boolean")
                return "boolean"
            else:
                print(f"{col} is categorical")
                return "categorical"
        elif isinstance(unique_values[0], float):
            if [unique_values[0], unique_values[1]] == [0.0, 1.0] or [
                unique_values[0],
                unique_values[1],
            ] == [1.0, 0.0]:
                print(f"{col} is boolean")
                return "boolean"
            else:
                print(f"{col} is categorical")
                return "categorical"

    print(f"{col} is categorical")

    return "categorical"


def _determine_column_type_int(df, col):
    """
    Case data frame column type is integer.
    It could either be "categorical", "boolean", or "continuous"

    :param df: Dataframe
    :param col: Current column being examined
    :return: Either "categorical", "boolean", or "continuous" value
    """
    values = df[col].unique()
    total_unique_values = df[col].value_counts().count()
    rows = df[col].count()
    bool_array_01 = numpy.array([0, 1])
    bool_array_10 = numpy.array([1, 0])

    if total_unique_values == 2 and (
        numpy.array_equal(values, bool_array_10)
        or (numpy.array_equal(values, bool_array_01))
    ):
        print(f"{col} is boolean")
        return "boolean"
    elif int(total_unique_values) / int(rows) <= 0.05:
        print(f"{col} is categorical")
        return "categorical"
    else:
        print(f"{col} is continuous")
        return "continuous"


def split_data_set(df, predictors):
    """
    Split dataset on predictors in list between categorical and continuous.

    :param df: Sample data frame that contains both p.
    :param predictors: List of predictor.
    :return: A tuple of (categorical predictors dataset, continuous predictor dataset)
    """
    categorical_predictors = []
    continuous_predictors = []
    col_type = "boolean"
    categorical_df = None
    continuous_df = None

    for col in predictors:
        # first, determine the column type, keep categorical and continuous columns.
        if df.dtypes[col] == "bool":
            print(f"{col} is boolean")
            col_type = "boolean"
        elif df.dtypes[col] == "float":
            print(f"{col} is continuous")
            col_type = "continuous"
        elif df.dtypes[col] == "category":
            col_type = _determine_column_type_category(df, col)
        elif df.dtypes[col] == "object":
            col_type = _determine_column_type_object(df, col)
        elif df.dtypes[col] in ("int32", "int64"):
            col_type = _determine_column_type_int(df, col)

        # then, add it to either categorical list or continuous list
        if col_type == "categorical" or col_type == "boolean":
            categorical_predictors.append(col)
        elif col_type == "continuous":
            continuous_predictors.append(col)

        categorical_df = df[categorical_predictors]
        continuous_df = df[continuous_predictors]

    return categorical_df, continuous_df


def add_cols_for_encoded_top_categories(df, cat_feature, top_categories):
    for category in top_categories:
        df[cat_feature + "_" + category] = numpy.where(
            df[cat_feature] == category, 1, 0
        )


#######################################
# Step 2: FEATURE INSPECTION (x vs. y)
#######################################
def create_heatmap(data_df, col, filename, plots_dir):
    conf_matrix = confusion_matrix(data_df[col], data_df["HomeTeamWins"])
    fig_heatmap = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_heatmap.update_layout(
        title="Categorical Predictor" + col + " by Categorical Response ",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    file_path_heatmap = os.path.join(plots_dir, filename)
    fig_heatmap.write_html(
        file=file_path_heatmap,
        include_plotlyjs="cdn",
    )


def create_distribution_plot(data_df, col, plots_dir, filename):
    group_labels = ["0", "1"]
    # Create distribution plot with custom bin_size
    N = data_df[data_df["HomeTeamWins"] == 0][col]
    Y = data_df[data_df["HomeTeamWins"] == 1][col]
    fig_1 = ff.create_distplot([N, Y], group_labels, bin_size=0.2)
    fig_1.update_layout(
        title=f"Continuous Predictor {col} by Categorical Response HomeTeamWins",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    file_path_distribution_plot = os.path.join(plots_dir, filename)
    fig_1.write_html(
        file=file_path_distribution_plot,
        include_plotlyjs="cdn",
    )


def cal_plot_pval_tscore(data_df, col, plots_dir):
    # Use logistic regression because the response is a boolean
    predictor = sm.add_constant(data_df[col])
    logit = sm.Logit(data_df["HomeTeamWins"], predictor)
    logit_fitted = logit.fit()

    # Get the stats
    t_value = round(logit_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logit_fitted.pvalues[1])
    filename = f"t_and_p_values_{col}.html"
    html_link = f"<a href='{filename}' target='_blank'> t_and_p_values_{col} </a>"

    # Plot the figure
    fig = px.scatter(x=data_df[col], y=data_df["HomeTeamWins"], trendline="ols")
    fig.update_layout(
        title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {col}",
        yaxis_title="y",
    )
    file_path_pval_tval = os.path.join(plots_dir, filename)
    fig.write_html(file=file_path_pval_tval, include_plotlyjs="cdn")
    return t_value, p_value, html_link


def cal_plot_dmr_uw_w(
    data_df, col, population_mean, plots_dir, bin_num=8
):  # reviewed 2
    """
    Calculate the Difference with Mean of Response. Similar to Mean Squared Errors.
    :param data_df:
    :param col:
    :param population_mean: mean of response y (HomeTeamWins in this case)
    :param bin_num: number of bins for each feature
    :param plots_dir: directory to hold the plots
    :return: Sum of difference with mean of response; Sum of weighted difference with mean of response; html_file
    """
    d1 = pandas.DataFrame(
        {
            "X": data_df[col],
            "y": data_df["HomeTeamWins"],
            "x_binned": pandas.cut(data_df[col], bin_num, labels=False),
        }
    )
    d2 = (
        d1.groupby(["x_binned"])
        .agg({"y": ["mean", "count"], "X": "mean"})
        .reset_index()
    )
    d2.columns = ["x_binned", "BinMean", "BinCount", "x_binned_mean"]

    d2["pop_mean"] = population_mean
    d2["pop_prop"] = d2.BinCount / len(data_df)

    d2["MSD"] = (d2.BinMean - d2.pop_mean) ** 2
    MSD_ranking = d2["MSD"].fillna(0).mean()

    d2["w_MSD"] = d2.MSD * d2["pop_prop"]
    w_MSD_ranking = d2["w_MSD"].sum() / bin_num

    # Plot difference with mean of response
    plt.clf()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=d2["x_binned_mean"], y=d2["BinCount"], name="Population"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=d2["x_binned_mean"],
            y=d2["BinMean"],
            line=dict(color="magenta"),
            name="BinMean",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=d2["x_binned_mean"],
            y=d2["pop_mean"],
            line=dict(color="red"),
            name="PopulationMean",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        height=600, width=800, title_text=f"Difference With Mean of Response {col}"
    )
    html_file = f"Diff_with_mean_of_response_{col}.html"
    file_path = os.path.join(plots_dir, html_file)
    fig.write_html(
        file=file_path,
        include_plotlyjs="cdn",
    )
    return MSD_ranking, w_MSD_ranking, html_file


def get_feature_importance(plots_dir, X, y):
    rf = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=1)
    rf.fit(X, y)
    importance = rf.feature_importances_
    plot_feature_importance(plots_dir, importance, X.columns, model="Random_Forest")

    return importance


def plot_feature_importance(plots_dir, importance, names, model):
    # Create arrays from feature importance and feature names
    feature_importance = numpy.array(importance)
    feature_names = numpy.array(names)

    # Create a DataFrame using a Dictionary
    data = {
        "feature_names": feature_names,
        "feature_importance": feature_importance,
    }
    fi_df = pandas.DataFrame(data)
    #
    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns_plot = sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    # Add chart labels
    plt.title(model + " Feature Importance")
    plt.xlabel("Plot Feature Importance")
    plt.ylabel("Plot Feature Importance")
    fig1 = sns_plot.get_figure()
    fig1.savefig(f"{plots_dir}/{model}_Feature_Importance.png")
    plt.clf()

    filename = f"{plots_dir}/{model}_Feature_Importance.html"

    with open(filename, "w") as file:
        file.write(
            "<h1>Random Forest Feature Importance </h1>\n"
            + f"<img src='{plots_dir}/{model}_Feature_Importance.png'"
            + "alt='Random Forest Feature Importance'>"
        )
        # file.write(f'<a href="{filename}" target="_blank"> "Random Forest Feature Importance" </a>')


################################
# Step 3: xi vs. xj CORRELATION
################################
def cal_cat_cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """(Prof's code)
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from:
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = numpy.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pandas.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cal_cat_cont_correlation(categories, values):
    """Prof's code
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pandas.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


# CORRELATION CONT-CONT
def create_table_and_matrix_cont_cont_correlation(plots_dir, continuous_df):
    if continuous_df.empty:
        print(
            "Continuous dataset is empty. Nothing to calculate continuous-continuous correlation."
        )
        return "N/A"

    data1_1 = []  # for table
    data1_2 = []  # for matrix

    for cont1 in continuous_df.columns.to_list():
        row_matrix = []
        for cont2 in continuous_df.columns.to_list():
            row_tbl = [cont1, cont2]
            cont1_array = continuous_df[cont1].to_numpy()
            cont2_array = continuous_df[cont2].to_numpy()
            pearson_r, p_value = stats.pearsonr(cont1_array, cont2_array)
            row_tbl.append(round(pearson_r, 3))
            data1_1.append(row_tbl)
            row_matrix.append(round(pearson_r, 3))
        data1_2.append(row_matrix)

    # convert data1_1 to dataframe (DESC order)
    print_subheading("Cont_Cont_Correlation_Ratio:")
    cont_cont_df_tbl = pandas.DataFrame(
        data1_1,
        columns=[
            "Continuous Predictor 1",
            "Continuous Predictor 2",
            "Pearson's Correlation Ratio",
        ],
    )
    cont_cont_df_tbl = cont_cont_df_tbl.sort_values(
        by=["Pearson's Correlation Ratio"], ascending=False
    )
    print(cont_cont_df_tbl.head())

    cont_cont_corr_tbl_html = cont_cont_df_tbl.to_html(escape=False, index=False)
    text_file = open(f"{plots_dir}/cont_cont_pearson_corr_table.html", "w")
    text_file.write(cont_cont_corr_tbl_html)
    text_file.close()

    # generate correlation matrix from data1_2
    print_subheading("Cont_Cont_Correlation_Ratio_Matrix:")
    cont_cont_df_matrix = pandas.DataFrame(
        data1_2,
        index=continuous_df.columns.to_list(),
        columns=continuous_df.columns.to_list(),
    )
    print(cont_cont_df_matrix.head())

    # visualize matrix
    fig1_cont_cont_pearson = ff.create_annotated_heatmap(
        z=data1_2,
        x=continuous_df.columns.to_list(),
        y=continuous_df.columns.to_list(),
        reversescale=True,
        showscale=True,
        zmin=-1,
        zmax=1,
    )
    fig1_cont_cont_pearson.update_layout(
        title="Continuous/ Continuous Correlation Matrix - Pearson's r"
    )
    # fig1_cont_cont_pearson.show()

    # save to output folder:
    filename1 = "cont_cont_pearson_corr_matrix.html"
    file_path1 = os.path.join(plots_dir, filename1)
    fig1_cont_cont_pearson.write_html(file=file_path1, include_plotlyjs="cdn")

    return filename1


# CORRELATION CONT-CAT
def create_table_and_matrix_cont_cat_correlation(
    plots_dir, categorical_df, continuous_df
):
    if categorical_df.empty or continuous_df.empty:
        print(
            "Either categorical dataset or continuous dataset is empty. Nothing to calculate continuous-categorical "
            "correlation."
        )
        return "N/A"

    data2_1 = []
    data2_2 = []

    for cont in continuous_df.columns.to_list():
        row_matrix = []
        for cat in categorical_df.columns.to_list():
            row_tbl = [cont, cat]
            cont_array = continuous_df[cont].to_numpy()
            cat_array = categorical_df[cat].to_numpy()
            eta = cal_cat_cont_correlation(cat_array, cont_array)
            row_tbl.append(eta)
            data2_1.append(row_tbl)
            row_matrix.append(eta)
        data2_2.append(row_matrix)

    # convert data2_1 to dataframe (DESC order)
    print_subheading("Cont_Cat_Correlation_Ratio:")
    cont_cat_df_tbl = pandas.DataFrame(
        data2_1,
        columns=[
            "Continuous Predictor",
            "Categorical Predictor",
            "ETA Correlation Ratio",
        ],
    )
    cont_cat_df_tbl = cont_cat_df_tbl.sort_values(
        by=["ETA Correlation Ratio"], ascending=False
    )
    print(cont_cat_df_tbl.head())

    cont_cat_corr_tbl_html = cont_cat_df_tbl.to_html(escape=False, index=False)
    text_file2 = open(f"{plots_dir}/cont_cat_eta_corr_table.html", "w")
    text_file2.write(cont_cat_corr_tbl_html)
    text_file2.close()

    # generate correlation matrix from data2_2
    print_subheading("Cont_Cat_Correlation_Ratio_Matrix:")
    cont_cat_df_matrix = pandas.DataFrame(
        data2_2,
        index=continuous_df.columns.to_list(),
        columns=categorical_df.columns.to_list(),
    )
    print(cont_cat_df_matrix.head())

    # visualize matrix
    x = categorical_df.columns.to_list()
    y = continuous_df.columns.to_list()
    fig2_cont_cat_eta = ff.create_annotated_heatmap(
        z=data2_2, x=x, y=y, reversescale=True, showscale=True, zmin=-1, zmax=1
    )
    fig2_cont_cat_eta.update_layout(
        title="Continuous/ Categorical Correlation Matrix - eta"
    )
    # fig2_cont_cat_eta.show()

    # save to output folder:
    filename2 = "cont_cat_eta_corr_matrix.html"
    file_path2 = os.path.join(plots_dir, filename2)
    fig2_cont_cat_eta.write_html(file=file_path2, include_plotlyjs="cdn")

    return filename2


# CORRELATION CAT-CAT
def create_table_and_matrix_cat_cat_correlation(plots_dir, categorical_df):
    if categorical_df.empty:
        print(
            "Categorical data set is empty. Nothing to calculate categorical-categorical correlation."
        )
        return "N/A", "N/A"

    data3_1 = []
    data3_2_cramersV = []
    data3_2_tschuprow = []

    for cat1 in categorical_df.columns.to_list():
        cramersV_matrix = []
        tschuprow_matrix = []
        for cat2 in categorical_df.columns.to_list():
            row_tbl = [cat1, cat2]
            cat1_array = categorical_df[cat1].to_numpy()
            cat2_array = categorical_df[cat2].to_numpy()
            cramersV = cal_cat_cat_correlation(x=cat1_array, y=cat2_array)
            tschuprow = cal_cat_cat_correlation(
                x=cat1_array, y=cat2_array, bias_correction=True, tschuprow=True
            )
            row_tbl.append(cramersV)
            row_tbl.append(tschuprow)
            data3_1.append(row_tbl)
            cramersV_matrix.append(cramersV)
            tschuprow_matrix.append(tschuprow)
        data3_2_cramersV.append(cramersV_matrix)
        data3_2_tschuprow.append(tschuprow_matrix)

    # convert data3_1 to dataframe (DESC order)
    print_subheading("Cat_Cat_Correlation_Ratio:")
    cat_cat_df_tbl = pandas.DataFrame(
        data3_1,
        columns=[
            "Categorical Predictor 1",
            "Categorical Predictor 2",
            "Cramer's V Correlation Ratio",
            "Tschuprow Correlation Ratio",
        ],
    )
    cat_cat_df_tbl = cat_cat_df_tbl.sort_values(
        by=["Cramer's V Correlation Ratio", "Tschuprow Correlation Ratio"],
        ascending=False,
    )
    print(cat_cat_df_tbl.head())

    cat_cat_corr_tbl_html = cat_cat_df_tbl.to_html(escape=False, index=False)
    text_file3 = open(f"{plots_dir}/cat_cat_cramersV_tschuprow_corr_table.html", "w")
    text_file3.write(cat_cat_corr_tbl_html)
    text_file3.close()

    # generate correlation matrix from data3_2_cramersV/tschuprow
    print_subheading("Cat_Cat_Correlation_Ratio_Matrix:")
    cat_cat_df_matrix_cramers = pandas.DataFrame(
        data3_2_cramersV,
        index=categorical_df.columns.to_list(),
        columns=categorical_df.columns.to_list(),
    )
    print("\ncat_cat_df_matrix_cramers:\n", cat_cat_df_matrix_cramers)
    cat_cat_df_matrix_tschuprow = pandas.DataFrame(
        data3_2_tschuprow,
        index=categorical_df.columns.to_list(),
        columns=categorical_df.columns.to_list(),
    )
    print("\ncat_cat_df_matrix_tschuprow:\n", cat_cat_df_matrix_tschuprow)

    # visualize matrix
    # CramersV
    fig3_cramers = ff.create_annotated_heatmap(
        z=data3_2_cramersV,
        x=categorical_df.columns.to_list(),
        y=categorical_df.columns.to_list(),
        reversescale=True,
        showscale=True,
        zmin=-1,
        zmax=1,
    )
    fig3_cramers.update_layout(
        title="Categorical/ Categorical Correlation Matrix - Cramer's V"
    )
    # fig3_cramers.show()

    # Tschuprow
    fig3_tschuprow = ff.create_annotated_heatmap(
        z=data3_2_tschuprow,
        x=categorical_df.columns.to_list(),
        y=categorical_df.columns.to_list(),
        reversescale=True,
        showscale=True,
        zmin=-1,
        zmax=1,
    )
    fig3_tschuprow.update_layout(
        title="Categorical/ Categorical Correlation Matrix - Tschuprow"
    )
    # fig3_tschuprow.show()

    # save to output folder:
    filename3_1 = "cat_cat_cramersV_corr_matrix.html"
    filename3_2 = "cat_cat_tschuprow_corr_matrix.html"
    file_path3_cramers = os.path.join(plots_dir, filename3_1)
    file_path3_tschuprow = os.path.join(plots_dir, filename3_2)
    fig3_cramers.write_html(file=file_path3_cramers, include_plotlyjs="cdn")
    fig3_tschuprow.write_html(file=file_path3_tschuprow, include_plotlyjs="cdn")

    return filename3_1, filename3_2


# CREATE TABLE WITH LINKS
def create_html_table_with_links(
    dataset_name,
    cont_cont_file,
    cont_cat_file,
    cat_cat_stchuprow_file,
    cat_cat_cramer_file,
    plots_dir,
):
    """Reference link: https://www.w3schools.com/html/html_tables.asp"""
    hyperlink = '<a href="{link}">{text}</a>'

    # what go to continuous row and continuous column
    if cont_cont_file != "N/A":
        cont_cont_text = hyperlink.format(link=cont_cont_file, text="Plot")
    else:
        cont_cont_text = "N/A"

    # what go to continuous row and categorical column
    if cont_cat_file != "N/A":
        cont_cat_text = hyperlink.format(link=cont_cat_file, text="Plot")
    else:
        cont_cat_text = "N/A"

    if cat_cat_stchuprow_file != "N/A":
        cat_cat_stchuprow_text = hyperlink.format(
            link=cat_cat_stchuprow_file, text="Plot"
        )
    else:
        cat_cat_stchuprow_text = "N/A"

    if cat_cat_cramer_file != "N/A":
        cat_cat_cramer_text = hyperlink.format(link=cat_cat_cramer_file, text="Plot")
    else:
        cat_cat_cramer_text = "N/A"

    content = f"""
    <!DOCTYPE html>
    <html>
    <style>
        table, th, td {{
          border:1px solid black;
        }}
    </style>
    <body>
    <h2>Predictors Correlation Plots for dataset {dataset_name}</h2>
    <!-- Create an html table that has links to the plots.
    +-------------+------------+-------------+
    |             | Continuous | Categorical |
    +-------------+------------+-------------+
    | Continuous  | LINK       | LINK        |
    | Categorical | LINK       | LINK        |
    +-------------+------------+-------------+
    -->
    <table>
    <tr>
        <th></th><th>Continuous</th><th>Categorical</th>
    </tr>
    <tr>
        <td>Continuous</td>
        <td>{cont_cont_text} </td>
        <td>{cont_cat_text}</td>
    </tr>
    <tr>
        <td>Categorical</td>
        <td>{cont_cat_text}</td>
        <td>Stchuprow's {cat_cat_stchuprow_text}<br>
        Cramer's {cat_cat_cramer_text}</td>
    </tr>
    </body>
    </html>
    """

    with open(f"{plots_dir}/Correlation_Table_{dataset_name}.html", "w") as f:
        f.write(content)


#####################################
# Step 4: BRUTE FORCE (xi, xj vs. y)
#####################################
# CALCULATE MSD AND WEIGHTED MSD
def cal_msd_w_uw(bin_num, binned1_gr, cont1, cont2, data1, population_mean):
    """
    :param bin_num: number of bins
    :param binned1_gr: dataframe
    :param cont1: continuous feature
    :param cont2: continuous feature
    :param data1: list to hold cell values to create dataframe later on
    :param population_mean: mean of response HomeTeamWins
    :return: mean_squared_diff_ranking, mean_squared_diff_weighted_ranking, dataframe binned1_gr
    """
    binned1_gr["residual"] = binned1_gr["BinMeans"] - population_mean
    binned1_gr["SquaredDiff"] = binned1_gr["residual"] ** 2
    # mean_squared_diff_ranking = binned1_gr["SquaredDiff"].sum() / bin_num
    mean_squared_diff_ranking = (
        binned1_gr["SquaredDiff"].fillna(0).mean()
    )  # -> correct to fix like this?

    # weigh the squared difference by population proportion
    bin_count_total = binned1_gr["BinCount"].sum()
    binned1_gr["population_proportion"] = binned1_gr["BinCount"] / bin_count_total
    binned1_gr["SquaredDiffWeighted"] = (
        binned1_gr["SquaredDiff"] * binned1_gr["population_proportion"]
    )
    mean_squared_diff_weighted_ranking = (
        binned1_gr["SquaredDiffWeighted"].sum() / bin_num
    )
    msd_tbl_row = [
        cont1,
        cont2,
        mean_squared_diff_ranking,
        mean_squared_diff_weighted_ranking,
    ]
    data1.append(msd_tbl_row)

    return mean_squared_diff_ranking, mean_squared_diff_weighted_ranking, binned1_gr


# CREATE MSD TABLE AND PLOT FOR CONT-CONT PREDICTORS
def create_msd_table_cont_cont(
    bin_num, continuous_df, data_set, population_mean, response, plots_dir
):
    cell_values_list = []
    plot_list = []

    # list to create a table for mean_squared_diff_ranking, mean_squared_diff_weighted_ranking
    for cont1 in continuous_df.columns.to_list():
        for cont2 in continuous_df.columns.to_list():
            x1_binned = pandas.cut(data_set[cont1], bin_num)
            x2_binned = pandas.cut(data_set[cont2], bin_num)
            dataset_binned1 = pandas.DataFrame(
                {
                    "x1": continuous_df[cont1],
                    "x2": continuous_df[cont2],
                    "x1_binned": x1_binned,
                    "x2_binned": x2_binned,
                    "y": data_set[response],
                }
            )
            binned1_gr = (
                dataset_binned1.groupby(["x1_binned", "x2_binned"])
                .agg({"y": ["mean", "count"]})
                .reset_index()
            )  # (reference: https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count
            # -mean-etc-using-pandas-groupby)
            binned1_gr.columns = ["x1_binned", "x2_binned", "BinMeans", "BinCount"]

            # mean squared difference
            _, _, df1 = cal_msd_w_uw(
                bin_num, binned1_gr, cont1, cont2, cell_values_list, population_mean
            )
            # plot
            if df1 is not None:
                z_data = df1.pivot(
                    index="x1_binned", columns="x2_binned", values="residual"
                )
                fig1 = go.Figure(data=[go.Surface(z=z_data.values)])

                fig1.update_layout(
                    title=f"Relationship b/w cont1 {cont1}, cont2 {cont2}, and Response - Residual (or Bin Difference)",
                    autosize=True,
                    scene=dict(
                        xaxis_title=f"Bins of cont2 {cont2}",
                        yaxis_title=f"Bins of cont1 {cont1}",
                        zaxis_title="Residual",
                    ),
                )
                filename_dmr_3d_cont_cont = f"BF_dmr_3d_cont_cont_{cont1}_{cont2}.html"
                filepath_dmr_3d_cont_cont = os.path.join(
                    plots_dir, filename_dmr_3d_cont_cont
                )
                fig1.write_html(file=filepath_dmr_3d_cont_cont, include_plotlyjs="cdn")
                plot_list.append(
                    f'<a href="{filename_dmr_3d_cont_cont}" target="_blank"> '
                    f"{filename_dmr_3d_cont_cont} </a>"
                )

    # convert data1 to dataframe (DESC order)
    print_subheading("Calculate the difference with the mean of response [cont-cont]: ")
    cont_cont_msd_tbl = pandas.DataFrame(
        cell_values_list,
        columns=[
            "Continuous Feature 1",
            "Continuous Feature 2",
            "MeanSquaredDiffRanking",
            "WeightedMeanSquaredDiffRanking",
        ],
    )
    cont_cont_msd_tbl["Plots (3D) for the Difference with Mean of Response"] = plot_list
    cont_cont_msd_tbl = cont_cont_msd_tbl.sort_values(
        by=["WeightedMeanSquaredDiffRanking"], ascending=False
    )
    print(cont_cont_msd_tbl.head())
    cont_cont_msd_tbl.sort_values(
        by=["WeightedMeanSquaredDiffRanking", "MeanSquaredDiffRanking"], ascending=False
    )
    cont_cont_msd_tbl_html = cont_cont_msd_tbl.to_html(escape=False, index=False)
    text_file_BF_1 = open(f"{plots_dir}/BF_dmr_w_uw_cont_cont_tbl.html", "w")
    text_file_BF_1.write(cont_cont_msd_tbl_html)
    text_file_BF_1.close()


# CREATE MSD TABLE AND PLOT CONT-CAT
def create_msd_table_cont_cat(
    bin_num,
    categorical_df,
    continuous_df,
    data_set,
    population_mean,
    response,
    plots_dir,
):
    data2 = []
    df2 = None
    plot_list = []

    for cont in continuous_df.columns.to_list():
        for cat in categorical_df.columns.to_list():
            cont_binned = pandas.cut(data_set[cont], bin_num)
            dataset_binned2 = pandas.DataFrame(
                {
                    "x1": continuous_df[cont],
                    "x2": categorical_df[cat],
                    "x1_binned": cont_binned,
                    "y": data_set[response],
                }
            )
            binned2_gr = (
                dataset_binned2.groupby(["x1_binned", "x2"])
                .agg({"y": ["mean", "count"]})
                .reset_index()
            )
            binned2_gr.columns = ["x1_binned", "x2", "BinMeans", "BinCount"]

            # mean squared difference
            _, _, df2 = cal_msd_w_uw(
                bin_num, binned2_gr, cont, cat, data2, population_mean
            )

            # plot
            if df2 is not None:
                z_data = df2.pivot(index="x1_binned", columns="x2", values="residual")
                fig2 = go.Figure(data=[go.Surface(z=z_data.values)])

                fig2.update_layout(
                    title=f"Relationship b/w cont {cont}, cat {cat} and Response - Residual (or Bin Difference)",
                    autosize=True,
                    scene=dict(
                        xaxis_title=f"Bins of cat {cat}",
                        yaxis_title=f"cont {cont}",
                        zaxis_title="Residual",
                    ),
                )
                filename_dmr_3d_cont_cat = f"BF_dmr_3d_cont_{cont}_cat_{cat}.html"
                filepath_dmr_3d_cont_cat = os.path.join(
                    plots_dir, filename_dmr_3d_cont_cat
                )
                fig2.write_html(file=filepath_dmr_3d_cont_cat, include_plotlyjs="cdn")
                plot_list.append(
                    f'<a href="{filename_dmr_3d_cont_cat}" target="_blank"> '
                    f"{filename_dmr_3d_cont_cat} </a>"
                )

    # convert data2 to dataframe (DESC order)
    print_subheading("Calculate the difference with the mean of response [cont-cat]: ")
    cont_cat_msd_tbl = pandas.DataFrame(
        data2,
        columns=[
            "Continuous Feature",
            "Categorical Feature",
            "MeanSquaredDiffRanking",
            "WeightedMeanSquaredDiffRanking",
        ],
    )
    cont_cat_msd_tbl["Plots (3D) for the Difference with Mean of Response"] = plot_list
    cont_cat_msd_tbl = cont_cat_msd_tbl.sort_values(
        by=["WeightedMeanSquaredDiffRanking"], ascending=False
    )
    print(cont_cat_msd_tbl.head())

    cont_cat_msd_tbl_html = cont_cat_msd_tbl.to_html(escape=False, index=False)
    text_file_BF_2 = open(f"{plots_dir}/BF_dmr_w_uw_cont_cat_tbl.html", "w")
    text_file_BF_2.write(cont_cat_msd_tbl_html)
    text_file_BF_2.close()


# CREATE MSD TABLE AND PLOT CAT-CAT
def create_msd_table_cat_cat(
    bin_num, categorical_df, data_set, population_mean, response, plots_dir
):
    data3 = []
    df3 = None
    plot_list = []

    for cat1 in categorical_df.columns.to_list():
        for cat2 in categorical_df.columns.to_list():
            dataset_binned3 = pandas.DataFrame(
                {
                    "x1": categorical_df[cat1],
                    "x2": categorical_df[cat2],
                    "y": data_set[response],
                }
            )
            binned3_gr = (
                dataset_binned3.groupby(["x1", "x2"])
                .agg({"y": ["mean", "count"]})
                .reset_index()
            )
            binned3_gr.columns = ["x1", "x2", "BinMeans", "BinCount"]

            # mean squared difference
            _, _, df3 = cal_msd_w_uw(
                bin_num, binned3_gr, cat1, cat2, data3, population_mean
            )
            # plot
            if df3 is not None:
                z_data = df3.pivot(index="x1", columns="x2", values="residual")
                fig3 = go.Figure(data=[go.Surface(z=z_data.values)])

                fig3.update_layout(
                    title=f"Relationship b/w cat1 {cat1}, cat2 {cat2} and Response - Residual (or Bin Difference)",
                    autosize=True,
                    scene=dict(
                        xaxis_title=f"cat2 {cat2}",
                        yaxis_title=f"cat1 {cat1}",
                        zaxis_title="Residual",
                    ),
                )
                filename_dmr_3d_cat_cat = f"BF_dmr_3d_cat1_{cat1}_cat2_{cat2}.html"
                filepath_dmr_3d_cat_cat = os.path.join(
                    plots_dir, filename_dmr_3d_cat_cat
                )
                fig3.write_html(file=filepath_dmr_3d_cat_cat, include_plotlyjs="cdn")
                plot_list.append(
                    f'<a href="{filename_dmr_3d_cat_cat}" target="_blank"> {filename_dmr_3d_cat_cat} </a>'
                )

    # convert data3 to dataframe (DESC order)
    print_subheading("Calculate the difference with the mean of response [cat-cat]: ")
    cat_cat_msd_tbl = pandas.DataFrame(
        data3,
        columns=[
            "Categorical Feature 1",
            "Categorical Feature 2",
            "MeanSquaredDiffRanking",
            "WeightedMeanSquaredDiffRanking",
        ],
    )
    cat_cat_msd_tbl["Plots (3D) for the Difference with Mean of Response"] = plot_list
    cat_cat_msd_tbl = cat_cat_msd_tbl.sort_values(
        by=["WeightedMeanSquaredDiffRanking"], ascending=False
    )
    print(cat_cat_msd_tbl.head())

    cat_cat_msd_tbl_html = cat_cat_msd_tbl.to_html(escape=False, index=False)
    text_file_BF_3 = open(f"{plots_dir}/BF_dmr_w_uw_cat_cat_tbl.html", "w")
    text_file_BF_3.write(cat_cat_msd_tbl_html)
    text_file_BF_3.close()
