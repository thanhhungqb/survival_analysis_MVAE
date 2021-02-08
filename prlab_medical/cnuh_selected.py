#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from prlab.common.utils import constant_map_dict
from prlab_medical.cnuh_constants import *
from prlab_medical.medicine_data_process import data_filter, data_preprocessing, data_transform


def select_data(fname, save_file):
    """
    prlab_medical-selected.csv
    :param fname:
    :return:
    """
    data = pd.read_csv(fname)
    # data

    header = ['Cancer Diagnosis', 'gender', 'Age at diagnosis', 'M-code1', 'M-code 2', 'M-code 3', 'Final weapon 1',
              'T / N / M weapon 1', 'Last Order Date1', 'Final weapon 2', 'T / N / M weapon 2', 'Last Order Date2',
              'Final Weapon 3', 'T / N / M Armory3', 'Last Order Date3', 'Smoking', 'Daily amount of smoking (A)',
              'Smoking period (years)', 'Non-smoking year', 'Survival', 'x']
    header = ['등록번호', 'gender', 'Cancer Diagnosis', '중증확진일', '사망일자(진단서 기준)', '진단서 등록일자', 'Age at diagnosis', 'M-code1',
              'Final weapon 1', 'T / N / M weapon 1', 'Last Order Date1', 'Smoking', 'Daily amount of smoking (A)',
              'Smoking period (years)', '흡연갑년', 'Non-smoking year', 'Survival', 'x(days)']
    data.columns = header
    used_col = ['Cancer Diagnosis', 'gender', 'Age at diagnosis', 'M-code1', 'Final weapon 1', 'T / N / M weapon 1',
                'Last Order Date1', 'Smoking', 'Daily amount of smoking (A)', 'Smoking period (years)',
                'Non-smoking year',
                'Survival']

    data = data[used_col]

    df = data_filter(data)
    df = data_preprocessing(df)
    df = data_transform(df)
    # df

    df['Survival_1'] = df['Survival'] - df['Age at diagnosis']

    df.to_csv(save_file, float_format='%.6f')

    # age at diagnosis and survival => year+month => convert to year,.. (month/12)
    # final weapon 1 (2/3) is 1A and IA is same (2A/IIA, 3A/IIIA, ...)
    # date format convert to continue value (day from 1970?)
    # gender M:0, F: 1, NA: 0.5?
    # why some row Survival M < Age at diagnosis


with open(MAP_FILE_NAME) as fp:
    jmap = json.load(fp=fp)
    constants = jmap['constants']
    # update based on cnuh_constants

    jmap = constant_map_dict(jmap)
    cnuh_map_name = jmap['cnuh_map_name']
    selected_header_en = jmap['selected_header']


def cnuh_data_transform(data_df, selected_header=None):
    """
    prlab_medical-selected.csv
    :param data_df: dataframe contains data
    :param selected_header:
    :return:
    """

    current_header = data_df.columns.tolist()
    mapped_header = [cnuh_map_name.get(k, k) for k in current_header]

    if selected_header is None:
        selected_header = selected_header_en
    data_df.columns = mapped_header + list(data_df.columns)[len(mapped_header):]

    df = data_preprocessing(data_df)
    df = data_transform(df)
    df = data_filter(df)

    df = df[selected_header]

    # df['Survival_1'] = df['Survival'] - df['Age at diagnosis']
    return df
