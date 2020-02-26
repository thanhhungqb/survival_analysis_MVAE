import numpy as np
import pandas as pd


def data_filter(df):
    """
    Filter data and keep only fields, values can be used
    """
    # remove all nan/empty Survival
    df = df[df['Survival'].notnull()]

    # remove all nan in Cancer Diagnosis / Age at diagnosis
    df = df[df['Cancer Diagnosis'].notnull()]
    df = df[df['Age at diagnosis'].notnull()]

    # remove all negative of Survival - Age at diagnosis (seem wrong data, several rows)
    df = df[df['Survival'] - df['Age at diagnosis'] > 0]

    # remove nan in 'M-code1' TODO
    df = df[df['M-code1'].notnull()]

    return df


def data_preprocessing(df):
    """
    preprocessing data including many steps: missing values, default values
    """
    # 'gender', 'Age at diagnosis', 'Cancer Diagnosis'
    df['gender'].fillna('NA', inplace=True)

    # 'M-code1'
    # data['M-code1'].fillna('UNK', inplace=True) # TODO not use

    # 'Final weapon 1', 'T / N / M weapon 1', 'Last Order Date1'
    # data['Final weapon 1'].fillna('UNK', inplace=True)
    # data['T / N / M weapon 1'].fillna('UNK', inplace=True)
    # data['Last Order Date1'].fillna('UNK', inplace=True)

    df['Smoking'].fillna('N', inplace=True)
    df['Non-smoking year'].fillna('UNK', inplace=True)  # TODO what value, death year?

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
    return df.dt.year + df.dt.month / 12 + df.dt.day / 12 / 30


def data_normalize(df, eps=0.01):
    """
    some normalize step
    """
    df.loc[:, 'Cancer Diagnosis'] = to_continue_d(df['Cancer Diagnosis'])
    df.loc[:, 'Last Order Date1'] = to_continue_d(df['Last Order Date1'])
    return df


def data_transform(df, eps=0.01):
    """
    transform data for easy process, including:
    - convert datatime
    - convert ordinal data to number
    - normalize data columns
    """
    # convert date from yyyymmdd to continue date (from 1970)
    df.loc[:, 'Cancer Diagnosis'] = pd.to_datetime(df['Cancer Diagnosis'], format='%Y%m%d')
    df.loc[:, 'Last Order Date1'] = pd.to_datetime(df['Last Order Date1'], format='%Y%m%d')
    # TODO datetime to count days

    # convert gender, Smoking to number (0/1)
    df.loc[:, 'gender'] = np.where(df['gender'] == 'M', 0, 1)
    df.loc[:, 'Smoking'] = np.where(df['Smoking'] == 'N', 0, 1)

    # convert None-smoking year
    # fix this column values
    df.loc[:, 'Non-smoking year'] = x_convert(df['Non-smoking year'])

    # back several years (some rows)
    tmp_x = df['Cancer Diagnosis'].dt.year.astype('int32') + df['Non-smoking year']
    df.loc[:, 'Non-smoking year'] = np.where(df['Non-smoking year'] >= 0, df['Non-smoking year'], tmp_x)

    # not stop until cancer
    n_n = df['Cancer Diagnosis'].dt.year.astype('int32') * df['Smoking']
    df.loc[:, 'Non-smoking year'] = np.where(df['Non-smoking year'] > 0, df['Non-smoking year'], n_n)

    # convert age to continue
    df.loc[:, 'Age at diagnosis'] = survival_time_norm(df['Age at diagnosis'])
    df.loc[:, 'Survival'] = survival_time_norm(df['Survival'])

    # for IA, IB, ... to 1A, 1B, 2A, ...
    lst1 = [('IA', '1A'), ('IB', '1B'), ('IIA', '2A'), ('IIB', '2B'),
            ('IIIA', '3A'), ('IIIB', '3B'), ('IVA', '4A'), ('IVB', '4B')]

    # for (xr, xn) in lst1:
    #     df['Final weapon 1'].replace(xr, xn, inplace=True)
    # df['Final weapon 1'] = 'S-' + df['Final weapon 1']
    df.loc[:, 'T / N / M weapon 1'] = np.where(df['T / N / M weapon 1'].str.strip() == '', np.nan,
                                               df['T / N / M weapon 1'])
    return data_normalize(df)


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
