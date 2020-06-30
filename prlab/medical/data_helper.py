from fastai.tabular import *

from prlab.gutils import encode_and_bind, column_map, clean_str, load_json_text_lines
from prlab.medical.cnuh_selected import cnuh_data_transform, selected_header_en, TNM_CODE_C, M_CODE_C, SURVIVAL_C

keep_m_code_lst = ['m8041/3', 'm8070/3', 'm8140/3']


def pre_process_pipe(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    do so pre-processing step in data-frame
    :param config: df
    :return: new config with new df
    """
    df = config['df']

    df = df[df['M-code(조직형)'].isin(keep_m_code_lst)].copy()
    print('data len', len(df))
    df = cnuh_data_transform(df, selected_header=selected_header_en + ['fold'])

    config['df'] = df

    return config


def df_read(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    :param config:
    :return:
    """
    pp = config['path'] / config['train_file']
    config['df'] = pd.read_excel(pp, sheet_name=config['sheet_name'])
    return config


def make_one_hot_df(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    Call before `prlab.medical.data_helper.data_load_df`
    Update some field in df and make config to work with one-hot
    :param config:
    :return:
    """
    ndf = encode_and_bind(config['df'], [TNM_CODE_C, M_CODE_C], keep_old=False)
    config['df'] = ndf

    # update cat_names to [] and cont_names to all fields (except fold)
    cont_names = config['df'].select_dtypes(include=[np.number]).columns.tolist()
    cont_names = [o for o in cont_names if o not in [SURVIVAL_C, config['dep_var']]]
    cont_names = [o for o in cont_names if o != 'fold']  # remove fold if has
    config['cat_names'], config['cont_names'] = [], cont_names

    return config


def make_embedding_df(**config):
    """
    from df and load embedding from file, make a new df and cont_names
    :param config:
    :return:
    """
    df = config['df']
    df.dropna(inplace=True)
    map_clean_str_rev = {clean_str(k): k for k in df.columns.tolist()}

    p = config['fold_weight']
    emb = load_json_text_lines(p)[config['fold']]
    emb = {map_clean_str_rev[k]: v for k, v in emb.items() if map_clean_str_rev.get(k, None) is not None}

    print('emb keys', emb.keys())
    lst = [TNM_CODE_C, M_CODE_C]
    ndf = column_map(config['df'], lst, emb, keep_old=False)
    config['df'] = ndf

    # update cat_names to [] and cont_names to all fields (except fold)
    cont_names = config['df'].select_dtypes(include=[np.number]).columns.tolist()
    cont_names = [o for o in cont_names if o not in [SURVIVAL_C, config['dep_var']]]
    cont_names = [o for o in cont_names if o != 'fold']  # remove fold if has
    config['cat_names'], config['cont_names'] = [], cont_names

    return config


def data_load_df(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    Make a data_train, data_test and add to config
    TODO fix some hard-code including procs, label_cls
    :param config:
    :return: new config
    """
    df = config['df']
    data_train_df = df[df['fold'] != config['test_fold']].copy()
    data_test_df = df[df['fold'] == config['test_fold']].copy()

    # cat_names = train_data.select_dtypes(include=['object']).columns.tolist()
    # cont_names = train_data.select_dtypes(include=[np.number]).columns.tolist()
    cat_names, cont_names = config['cat_names'], config['cont_names']

    procs = [FillMissing, Categorify, Normalize][:2]

    # Test tabularlist
    test = TabularList.from_df(data_test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)

    # Train data bunch
    data_train = (
        TabularList.from_df(data_train_df, path=config['path'], cat_names=cat_names, cont_names=cont_names, procs=procs)
            .split_by_rand_pct(valid_pct=0.1, seed=config.get('seed', 43))
            .label_from_df(cols=config['dep_var'], label_cls=FloatList, log=config['is_log'])
            .add_test(test)
            .databunch())

    print(data_train.show_batch(rows=10))
    print(data_train)

    config.update({
        'data_train': data_train,
        'data_test': data_test_df
    })

    return config


# ------------- new version of data 2020-05 --------------------------
# PatientID	gender	age	Survival.time	Mcode
# Mcode.description	Histology	Overall.stage
# Clinical.T.Stage	Clinical.N.stage	Clinical.M.stage
# Smoking.status	Smoking.amount
# Deadstatus.event
# PatientWeight	PatientSize

class XConst:
    GENDER_C = "gender"
    AGE_AT_DIAGNOSIS_C = "age"
    M_CODE_C = "Mcode"
    M_CODE_DESC_C = "Mcode.description"
    HIS_C = "Histology"

    STAGE_CODE_C = "Overall.stage"
    T_STAGE_C = "Clinical.T.Stage"
    N_STAGE_C = "Clinical.N.stage"
    M_STAGE_C = "Clinical.M.stage"

    SMOKING_C = "Smoking.status"
    SMOKING_TOTAL_C = 'Smoking.amount'

    DEAD_STATUS_C = "Deadstatus.event"
    WEIGHT_C = "PatientWeight"
    SIZE_C = "PatientSize"

    SURVIVAL_C = 'Survival.time'
    NOTE_C = 'note'

    DEAD_STATUS_V = 1


class SimpleCNUHPreProcessing:
    _xconst = XConst

    def __init__(self, **config):
        self.config = config

        self.selected_header = config.get('selected_header', None)

    def __call__(self, **config):
        df = config['df']
        df = self.filter_pre(df)
        df = self.transform(df)
        df = self.filter_post(df)
        df = self.keep_header(df)

        config['df'] = df
        return config

    def filter_pre(self, df):
        """
        Filter data before other steps
        :param df:
        :return:
        """
        return df

    def filter_post(self, df):
        """
        filter data frame after other steps
        :param df:
        :return:
        """
        # remove unsure case
        df = df[df[self._xconst.WEIGHT_C].notnull()]
        df = df[df[self._xconst.SIZE_C].notnull()]
        df = df[df[self._xconst.NOTE_C].isnull()] if self._xconst.NOTE_C in list(df.columns) else df

        # use alive and discontinue cases?
        # df = df[df[self._xconst.DEAD_STATUS_C] == self._xconst.DEAD_STATUS_V]

        return df

    def keep_header(self, df):
        if self.selected_header is not None:
            df = df[self.selected_header]
        return df

    def transform(self, df):
        # gender
        df.loc[:, self._xconst.GENDER_C] = np.where(df[self._xconst.GENDER_C] == 'male', 0, 1)

        return df
