import numpy as np
import pandas as pd

# define some constants
from prlab_medical.cnuh_constants import *


def data_filter(df):
    """
    Filter data and keep only fields, values can be used
    """
    # remove all nan/empty Survival
    df = df[df[SURVIVAL_C].notnull()]

    # remove all nan in Cancer Diagnosis / Age at diagnosis
    df = df[df[CANCER_DIAGNOSIS_C].notnull()]
    df = df[df[AGE_AT_DIAGNOSIS_C].notnull()]

    # remove all negative of Survival - Age at diagnosis (seem wrong data, several rows)
    df = df[df[SURVIVAL_C] - df[AGE_AT_DIAGNOSIS_C] > 0]

    # remove nan in 'M-code1' TODO
    df = df[df[M_CODE_C].notnull()]

    return df


def data_preprocessing(df):
    """
    preprocessing data including many steps: missing values, default values
    """
    # 'gender', 'Age at diagnosis', 'Cancer Diagnosis'
    df[GENDER_C].fillna('NA', inplace=True)

    # 'M-code1'
    # data['M-code1'].fillna('UNK', inplace=True) # TODO not use

    # 'Final weapon 1', 'T / N / M weapon 1', 'Last Order Date1'
    # data['Final weapon 1'].fillna('UNK', inplace=True)
    # data['T / N / M weapon 1'].fillna('UNK', inplace=True)
    # data['Last Order Date1'].fillna('UNK', inplace=True)

    df[SMOKING_C].fillna('N', inplace=True)
    df[NON_SMOKING_YEAR_C].fillna('UNK', inplace=True)  # TODO what value, death year?

    return df


def survival_time_norm(df_col, eps=0.01):
    """
    convert from old.mm to old.(mm/12) to continous value
    """
    n = (df_col * 100 + eps).astype('int32')
    p_y = n // 100
    p_m = n % 100  # p_m < 12
    n_s = p_y + p_m / 12
    return n_s


def time_norm_to_survival(df_col, eps=0.01):
    """
    reverse version of `survival_time_norm`
    """
    y = df_col.astype('int32')
    m = ((df_col - y) * 12 + eps).astype('int32')
    n_col = y + m / 100
    return n_col


def fix_non_smoking_year(df_col):
    """
    fix Non-smoking year from many typy to year only and convert to int32
    types: x, 07, x year ago, %y%m, %y.%m(.), %y.%mm
    """


def to_continue_d(df):
    return df.dt.year + df.dt.month / 12 + df.dt.day / 365


def data_normalize(df, eps=0.01):
    """
    some normalize step
    """
    df.loc[:, CANCER_DIAGNOSIS_C] = to_continue_d(df[CANCER_DIAGNOSIS_C])
    df.loc[:, LAST_ORDER_DATE_C] = to_continue_d(df[LAST_ORDER_DATE_C])
    return df


def data_transform(df, eps=0.01):
    """
    transform data for easy process, including:
    - convert datatime
    - convert ordinal data to number
    - normalize data columns
    """
    # convert date from yyyymmdd to continue date (from 1970)
    df.loc[:, CANCER_DIAGNOSIS_C] = pd.to_datetime(df[CANCER_DIAGNOSIS_C], format='%Y%m%d')
    df.loc[:, LAST_ORDER_DATE_C] = pd.to_datetime(df[LAST_ORDER_DATE_C], format='%Y%m%d')
    # TODO datetime to count days

    # convert gender, Smoking to number (0/1)
    df.loc[:, GENDER_C] = np.where(df[GENDER_C] == 'M', 0, 1)
    df.loc[:, SMOKING_C] = np.where(df[SMOKING_C] == 'N', 0, 1)

    # convert None-smoking year
    # fix this column values
    df.loc[:, NON_SMOKING_YEAR_C] = x_convert(df[NON_SMOKING_YEAR_C])

    # back several years (some rows)
    tmp_x = df[CANCER_DIAGNOSIS_C].dt.year.astype('int32') + df[NON_SMOKING_YEAR_C]
    df.loc[:, NON_SMOKING_YEAR_C] = np.where(df[NON_SMOKING_YEAR_C] >= 0, df[NON_SMOKING_YEAR_C], tmp_x)

    # not stop until cancer
    n_n = df[CANCER_DIAGNOSIS_C].dt.year.astype('int32') * df[SMOKING_C]
    df.loc[:, NON_SMOKING_YEAR_C] = np.where(df[NON_SMOKING_YEAR_C] > 0, df[NON_SMOKING_YEAR_C], n_n)

    # convert age to continue
    # df.loc[:, 'Age at diagnosis'] = survival_time_norm(df['Age at diagnosis'])
    df.loc[:, AGE_AT_DIAGNOSIS_C] = df[AGE_AT_DIAGNOSIS_C] / 365
    # df.loc[:, 'Survival'] = survival_time_norm(df['Survival'])
    df.loc[:, SURVIVAL_C] = df[SURVIVAL_C] / 365

    # for IA, IB, ... to 1A, 1B, 2A, ...
    lst1 = [('IA', '1A'), ('IB', '1B'), ('IIA', '2A'), ('IIB', '2B'),
            ('IIIA', '3A'), ('IIIB', '3B'), ('IVA', '4A'), ('IVB', '4B')]

    # for (xr, xn) in lst1:
    #     df['Final weapon 1'].replace(xr, xn, inplace=True)
    # df['Final weapon 1'] = 'S-' + df['Final weapon 1']
    df.loc[:, TNM_CODE_C] = np.where(df[TNM_CODE_C].str.strip() == '', np.nan, df[TNM_CODE_C])

    # normalize
    df = data_normalize(df)

    # add more field
    df[DELAY_TEST_C] = df[LAST_ORDER_DATE_C] - df[CANCER_DIAGNOSIS_C]
    df.loc[:, X_SURVIVAL_C] = df[X_SURVIVAL_C] / 365

    return df


def x_convert(lst):
    nlst = [pd.to_datetime(o, format='%Y.%m', errors='ignore') for o in lst]
    nlst = [pd.to_datetime(o[:-1] if o.endswith('.') else o, format='%Y.%m', errors='ignore')
            if not isinstance(o, pd.Timestamp) else o for o in nlst]
    nlst = [pd.to_datetime(o, format='%Y%m', errors='ignore')
            if not isinstance(o, pd.Timestamp) else o for o in nlst]
    nlst = [(int("-" + o[:-2]) if o.endswith('년전') else 0)
            if not isinstance(o, pd.Timestamp) else o for o in nlst]

    # get year only
    nlst = [o.year if isinstance(o, pd.Timestamp) else o for o in nlst]

    return nlst
