import pandas as pd
import os
import re
import ast
import numpy as np


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


def func(lst, metr):
    if isinstance(lst, float):
        return None
    if isinstance(lst, str):
        lst = ast.literal_eval(lst)
    series = pd.Series(lst)
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


def calc_one_user(dir_path):
    dir_data = folder_to_dataframe_str(dir_path)
    dir_data_nc = calc_all_cells(dir_data, func)
    return dir_data_nc[[
        "stand_rrg_sdnn", "lying_rrg_sdnn", "d_sdnn",
        "stand_rrg_rmssd", "lying_rrg_rmssd", "d_rmssd",
        "stand_rrg_nn50", "lying_rrg_nn50", "d_nn50",
        "stand_rrg_pnn50", "lying_rrg_pnn50", "d_pnn50",
    ]].T
# print(os.listdir())
print(calc_one_user("Arseny/folder"))
# calc_one_group()compare_groups()
