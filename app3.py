import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialog,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


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


def folder_to_dataframe_int(dir_path):
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
        n, typ, ext = stardart_file_name_int(file)
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            if (data.n == n).any():
                if data.loc[data["n"] == n, typ + "_" + ext].isna().all():
                    data.at[data.index[data["n"] == n].tolist()[0], typ + "_" + ext] = (
                        file_to_list(file_path)
                    )
                else:
                    data.at[data.index[data["n"] == n].tolist()[0], typ + "_" + ext] = (
                        data.at[data.index[data["n"] == n].tolist()[0], typ + "_" + ext]
                        + file_to_list(file_path)
                    )
            else:
                data.at[count_lines, typ + "_" + ext] = file_to_list(file_path)
                data.at[count_lines, "n"] = n
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
    return (data[cols[0]] - data[cols[1]]).abs().round(4)


def calc_all_cells(data, func):
    metr = ["sdnn", "rmssd", "nn50", "pnn50"]
    typ = ["stand_rrg", "lying_rrg", "stand_rrn", "lying_rrn"]
    for i in typ:
        for j in metr:
            data[i + "_" + j] = data[i].map(lambda x: round(func(x, j), 4))
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
    rr_s = make_series(series[0])
    rr_s_next = rr_s.shift(-1)
    rr = rr_s.values
    rr_next = rr_s_next.values[:-1]
    sd1 = (2**0.5) * pd.Series(rr_next - rr[:-1]).std()
    sd2 = (2**0.5) * pd.Series(rr_next + rr[:-1]).std()
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=rr[:-1], y=rr_next, alpha=0.5)
    plt.plot([min(rr), max(rr)], [min(rr), max(rr)], color="red")
    plt.xlabel("RR(n), мс")
    plt.ylabel("RR(n+1), мс")
    plt.title(f"Пуанкаре плот RR интервалов для {name}")
    plt.savefig(f"{name}_{typ}.png")


def calc_user(dir_path):
    dir_data = folder_to_dataframe_int(dir_path)
    dir_data_nc = calc_all_cells(dir_data, func)
    if len(dir_data_nc[cols]) == 1:
        if dir_data["stand_rrg"][0] != None:
            make_paun_plot(dir_data["stand_rrg"], dir_path, "stand_rrg")
        if dir_data["lying_rrg"][0] != None:
            make_paun_plot(dir_data["lying_rrg"], dir_path, "lying_rrg")
    return dir_data_nc.set_index("n")[cols].T


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
                    t_list.append([col, data1[0], data2[0], data1[1][col].dropna().median(), data2[1][col].dropna().median(), round(p_val, 4)])
    return pd.DataFrame(t_list, columns=["metric", "gr1", "gr2", "med_gr1", "med_gr2", "p-val"]).set_index(
        "metric"
    )



def display_dataframe(data: pd.DataFrame):
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    table_widget = QTableWidget()
    table_widget.setRowCount(data.shape[0])
    table_widget.setColumnCount(data.shape[1])
    table_widget.setHorizontalHeaderLabels(data.columns.astype(str))
    table_widget.setVerticalHeaderLabels(data.index.astype(str))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            item = QTableWidgetItem(str(data.iat[i, j]))
            item.setFlags(
                item.flags() & ~Qt.ItemIsEditable
            )  # Make the item non-editable
            table_widget.setItem(i, j, item)

    table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table_widget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    layout.addWidget(table_widget)
    dialog.setLayout(layout)
    dialog.setWindowTitle("DataFrame Viewer")
    dialog.resize(1200, 600)
    dialog.exec_()


def display_image(image_path):
    dialog = QDialog()
    layout = QVBoxLayout(dialog)

    label = QLabel(dialog)
    pixmap = QPixmap(image_path)
    label.setPixmap(pixmap)
    layout.addWidget(label)

    dialog.setLayout(layout)
    dialog.setWindowTitle("Graph Viewer")
    dialog.exec_()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analysis Application")

        self.user_button = QPushButton("User Analysis")
        self.group_button = QPushButton("Group Analysis")
        self.compare_button = QPushButton("Group Comparison")

        layout = QVBoxLayout()
        layout.addWidget(self.user_button)
        layout.addWidget(self.group_button)
        layout.addWidget(self.compare_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.user_button.clicked.connect(self.perform_user_analysis)
        self.group_button.clicked.connect(self.perform_group_analysis)
        self.compare_button.clicked.connect(self.perform_group_comparison)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        return folder_path

    def perform_user_analysis(self):
        folder_path = self.select_folder()
        if folder_path:
            user_data = calc_user(folder_path)

            if os.path.exists(f"{folder_path}_stand_rrg.png"):
                stand_dialog = QDialog(self)
                stand_layout = QVBoxLayout(stand_dialog)
                stand_label = QLabel(stand_dialog)
                stand_pixmap = QPixmap(f"{folder_path}_stand_rrg.png")
                stand_label.setPixmap(stand_pixmap)
                stand_layout.addWidget(stand_label)
                stand_dialog.setWindowTitle("Stand RRG Image")
                stand_dialog.show()

            if os.path.exists(f"{folder_path}_lying_rrg.png"):
                lying_dialog = QDialog(self)
                lying_layout = QVBoxLayout(lying_dialog)
                lying_label = QLabel(lying_dialog)
                lying_pixmap = QPixmap(f"{folder_path}_lying_rrg.png")
                lying_label.setPixmap(lying_pixmap)
                lying_layout.addWidget(lying_label)
                lying_dialog.setWindowTitle("Lying RRG Image")
                lying_dialog.show()

            dialog = QDialog(self)
            layout = QVBoxLayout(dialog)
            table_widget = QTableWidget()
            table_widget.setRowCount(user_data.shape[0])
            table_widget.setColumnCount(user_data.shape[1])
            table_widget.setHorizontalHeaderLabels(user_data.columns.astype(str))
            table_widget.setVerticalHeaderLabels(user_data.index.astype(str))

            for i in range(user_data.shape[0]):
                for j in range(user_data.shape[1]):
                    item = QTableWidgetItem(str(user_data.iat[i, j]))
                    item.setFlags(
                        item.flags() & ~Qt.ItemIsEditable
                    )  # Make the item non-editable
                    table_widget.setItem(i, j, item)

            table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table_widget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(table_widget)
            dialog.setLayout(layout)
            dialog.setWindowTitle("DataFrame Viewer")
            dialog.resize(1200, 600)
            dialog.show()

    def perform_group_analysis(self):
        folder_path = self.select_folder()
        if folder_path:
            group_data = calc_one_group(folder_path)
            display_dataframe(group_data)

    def perform_group_comparison(self):
        folder_path = self.select_folder()
        if folder_path:
            comparison_data = compare_groups(folder_path)
            display_dataframe(comparison_data)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
