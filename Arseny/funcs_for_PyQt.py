import pandas as pd
import os
import re
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import ttest_ind

cols = [
    "stand_rrg_sdnn",
    "stand_rrg_rmssd",
    "stand_rrg_nn50",
    "stand_rrg_pnn50",
    "lying_rrg_sdnn",
    "lying_rrg_rmssd",
    "lying_rrg_nn50",
    "lying_rrg_pnn50",
    "d_sdnn",
    "d_rmssd",
    "d_nn50",
    "d_pnn50",
]


def file_to_list(file_path):
    return (
        pd.read_csv(file_path, skiprows=2, header=None, encoding="utf-8")
        .iloc[:, 0]
        .tolist()
    )


def stardart_file_name_int(name):
    name = name.lower()
    n = int(re.search(r"\d+", name).group())
    typ = "stand" if "стоя" in name else "lying"
    ext = "rrn" if "rrn" in name else "rrg"
    return n, typ, ext


def standard_file_name_str(name):
    name = name.lower()
    n = str(re.search(r"\d+", name).group())
    typ = "stand" if "стоя" in name else "lying"
    ext = "rrn" if "rrn" in name else "rrg"
    return n, typ, ext


def folder_to_dataframe_str(dir_path):
    files = os.listdir(dir_path)
    nones = [None] * len(files)
    data = pd.DataFrame(
        {
            "n": nones,
            "stand_rrg": nones,
            "lying_rrg": nones,
            "stand_rrn": nones,
            "lying_rrn": nones,
        }
    )
    for count_lines, file in enumerate(files):
        n, typ, ext = standard_file_name_str(file)
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            if (data["n"] == n).any():
                row_index = data.index[data["n"] == n].tolist()[0]
                if data.at[row_index, typ + "_" + ext] is None:
                    data.at[row_index, typ + "_" + ext] = file_to_list(file_path)
                else:
                    data.at[row_index, typ + "_" + ext] += file_to_list(file_path)
            else:
                data.at[count_lines, "n"] = n
                data.at[count_lines, typ + "_" + ext] = file_to_list(file_path)
    return data.dropna(how="all")


def make_series(lst):
    if isinstance(lst, str):
        lst = ast.literal_eval(lst)
    if isinstance(lst, float):
        return None
    return pd.Series(lst)


def func(lst, metr):
    series = make_series(lst)
    series = series[series >= 0]
    if metr == "sdnn":
        return series.std()
    elif metr == "rmssd":
        return np.sqrt(np.mean(series.diff() ** 2))
    elif metr == "nn50":
        return np.sum(series.diff().abs() > 0.05)
    elif metr == "pnn50":
        return (np.sum(series.diff().abs() > 0.05)) / (len(series) - 1) * 100
    else:
        return None


def dif_stand_lying(data, cols):
    return (data[cols[0]] - data[cols[1]]).abs()


def calc_all_cells(data, func):
    metr = ["sdnn", "rmssd", "nn50", "pnn50"]
    typ = ["stand_rrg", "lying_rrg", "stand_rrn", "lying_rrn"]
    for i in typ:
        for j in metr:
            data[i + "_" + j] = data[i].map(lambda x: func(x, j))
    for i in [
        ["stand_rrg_sdnn", "lying_rrg_sdnn", "d_sdnn"],
        ["stand_rrg_rmssd", "lying_rrg_rmssd", "d_rmssd"],
        ["stand_rrg_nn50", "lying_rrg_nn50", "d_nn50"],
        ["stand_rrg_pnn50", "lying_rrg_pnn50", "d_pnn50"],
    ]:
        data[i[2]] = dif_stand_lying(data, i)
    return data


def calc_median_cols(data, cols):
    return [data[i].median() if not data[i].isna().all() else np.nan for i in cols]


def normal_means(
    s, choice_iterations=10000, sample_size=100, shapiro_iterations=50
) -> float:

    unique_elems = s.unique()
    arr = pd.Series(
        [
            np.random.choice(unique_elems, len(s)).mean()
            for _ in range(choice_iterations)
        ]
    )
    return np.mean(
        [shapiro(arr.sample(sample_size))[1] for _ in range(shapiro_iterations)]
    )


def has_zero_range(data):
    return data.max() - data.min() == 0


def make_t_table(s1, s2):
    if has_zero_range(s1) or has_zero_range(s2):
        return None
    if (
        (normal_means(s1) >= 0.05)
        and (normal_means(s2) >= 0.05)
        and (len(s1.unique()) != 1)
        and (len(s2.unique()) != 1)
    ):
        return ttest_ind(s1, s2, equal_var=(levene(s1, s2)[-1] >= 0.5))[-1]
    else:
        return None


def make_paun_plot(series, name, typ):
    # print(type(series), series)
    rr_s = make_series(series[0])
    rr_s_next = rr_s.shift(-1)
    rr = rr_s.values
    rr_next = rr_s_next.values[:-1]
    sd1 = (2**0.5) * pd.Series(rr_next - rr[:-1]).std()
    sd2 = (2**0.5) * pd.Series(rr_next + rr[:-1]).std()
    print(rr[:-1])
    print(rr_next)
    plt.figure(figsize=(8,8))
    sns.scatterplot(x=rr[:-1], y=rr_next, alpha=0.5)
    plt.plot([min(rr), max(rr)], [min(rr), max(rr)], color="red")
    plt.xlabel("RR(n), мс")
    plt.ylabel("RR(n+1), мс")
    plt.title(f"Пуанкаре плот RR интервалов для {name}")
    plt.savefig(f'{name}_{typ}.png')

def calc_user(dir_path):
    dir_data = folder_to_dataframe_str(dir_path)
    dir_data_nc = calc_all_cells(dir_data, func)
    if len(dir_data_nc[cols]) == 1:
        if dir_data["stand_rrg"][0] != None:
            make_paun_plot(dir_data["stand_rrg"], dir_path, "stand_rrg")
        if dir_data["lying_rrg"][0] != None:
            make_paun_plot(dir_data['lying_rrg'], dir_path, "lying_rrg")
    return dir_data_nc[cols].T


def calc_one_group(dir_path):
    dir_data = calc_user(dir_path).T
    return pd.DataFrame(
        {"group_data": calc_median_cols(dir_data, cols)},
        index=cols,
    )


def compare_groups(dir_path):
    all_groups = [
        [path, calc_user(os.path.join(dir_path, path)).T]
        for path in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, path))
    ]
    t_list = []
    for i, data1 in enumerate(all_groups):
        for j, data2 in enumerate(all_groups):
            c = [
                "stand_rrg_sdnn",
                "stand_rrg_rmssd",
                "stand_rrg_pnn50",
                "lying_rrg_sdnn",
                "lying_rrg_rmssd",
                "lying_rrg_pnn50",
                "d_sdnn",
                "d_rmssd",
                "d_pnn50",
            ]
            for col in c:
                if data1[1][col].isna().all() or data2[1][col].isna().all() or (i <= j):
                    continue
                p_val = make_t_table(data1[1][col].dropna(), data2[1][col].dropna())
                if not p_val:
                    continue
                if p_val <= 0.05:
                    t_list.append([col, data1[0], data2[0], p_val])
    return pd.DataFrame(t_list, columns=["metric", "gr1", "gr2", "p-val"]).set_index(
        "metric"
    )


print(compare_groups("data"))














































