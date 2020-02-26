#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from prlab.medical.medicine_data_process import data_filter, data_preprocessing, data_transform


def select_data(fname, save_file):
    """
    medical-selected.csv
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


header_kr = [
    '등록번호', '성별', '중증등록일', '중증확진일', '사망일자(진단서 기준)', '진단서 등록일자', '진단시나이(진단시 생존일수)',
    'M-code(조직형)', '최종병기', 'T/N/M병기', '최종병기날짜', '흡연력', '하루흡연량(갑)', '흡연기간(년)', '흡연갑년',
    '금연한 연도', '생존일수', 'x(days)', 'fold']
header_en = [
    '등록번호', 'gender', 'Cancer Diagnosis', '중증확진일', '사망일자(진단서 기준)', '진단서 등록일자', 'Age at diagnosis',
    'M-code1', 'Final weapon 1', 'T / N / M weapon 1', 'Last Order Date1', 'Smoking',
    'Daily amount of smoking (A)', 'Smoking period (years)', 'Smoking total (rel)', 'Non-smoking year', 'Survival',
    'x(days)', 'fold']

cnuh_map_name = {header_en[i]: header_kr[i] for i in range(len(header_kr))}

selected_header_en = [
    'M-code1', 'T / N / M weapon 1', 'Cancer Diagnosis', 'gender', 'Age at diagnosis',
    'Last Order Date1', 'Smoking', 'Daily amount of smoking (A)', 'Smoking period (years)', 'Smoking total (rel)',
    'Non-smoking year', 'Survival', 'x(days)']
selected_header_kr = [
    'M-code(조직형)', 'T/N/M병기', '성별', '중증등록일', '진단시나이(진단시 생존일수)', '최종병기날짜',
    '흡연력', '하루흡연량(갑)', '흡연기간(년)', '흡연갑년', '금연한 연도', '생존일수', 'x(days)']


def cnuh_data_transform(data, header=None, selected_header=None):
    """
    medical-selected.csv
    :param fname:
    :return:
    """

    if header is None:
        header = header_en

    if selected_header is None:
        selected_header = selected_header_en
    data.columns = header + list(data.columns)[len(header):]
    data = data[selected_header]

    df = data_preprocessing(data)
    df = data_transform(df)
    df = data_filter(df)
    # df

    # df['Survival_1'] = df['Survival'] - df['Age at diagnosis']
    return df

    # age at diagnosis and survival => year+month => convert to year,.. (month/12)
    # final weapon 1 (2/3) is 1A and IA is same (2A/IIA, 3A/IIIA, ...)
    # date format convert to continue value (day from 1970?)
    # gender M:0, F: 1, NA: 0.5?
    # why some row Survival M < Age at diagnosis

# pp = Path('/ws/data/cnuh')
# select_data(pp / 'file-2.csv', pp / 'file-processed.csv')
