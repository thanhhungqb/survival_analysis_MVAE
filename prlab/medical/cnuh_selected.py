#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

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
              'Smoking period (years)', '흡연갑년', 'Non-smoking year', 'Survival', 'x(days)'
              ]
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


pp = Path('/ws/data/cnuh')
select_data(pp / 'file-2.csv', pp / 'file-processed.csv')
