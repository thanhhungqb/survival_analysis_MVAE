import deprecation
from fastai.tabular import *
from sklearn.model_selection import train_test_split

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


class DfRateKeepFilter:
    """
    Follow Pipe template.
    Work with df in config and filter to keep only rate
    """

    def __init__(self, rate_keep, seed=None, **config):
        """

        :param rate_keep: (0.0, 1.0)
        :param seed: None or a number
        :param config:
        """
        self.rate_keep = rate_keep
        self.seed = seed
        assert 0 < self.rate_keep < 1

    def __call__(self, *args, **config):
        df = config['df'].reset_index(drop=True)
        idx = list(df.index)
        if self.seed is not None:
            np.random.seed(self.seed)

        selected = np.random.choice(idx, size=int(self.rate_keep * len(idx)))
        selected = selected.tolist()

        selected_df = df.iloc[selected].copy().reset_index(drop=True)
        config['df'] = selected_df
        return config


class DfRateKeepTrainFilter(DfRateKeepFilter):
    """
    Similar to `DfRateKeepFilter` except keep the original test_fold if have
    """

    def __init__(self, rate_keep, seed=None, **config):
        super(DfRateKeepTrainFilter, self).__init__(rate_keep, seed, **config)

    def __call__(self, *args, **config):
        df = config['df']
        train_df = df[df['fold'] != config['test_fold']]
        test_df = df[df['fold'] == config['test_fold']]
        config['df'] = train_df
        config = super().__call__(*args, **config)

        config['df'] = pd.concat([config['df'], test_df]).reset_index(drop=True)

        return config


@deprecation.deprecated(details="prlab.fastai.pipeline.make_one_hot_df_pipe is more general and consider to use")
def make_one_hot_df(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    Consider to use `prlab.fastai.pipeline.make_one_hot_df_pipe`, two pipe have similar idea but different implement
    Call before `prlab.medical.data_helper.data_load_df`
    Update some field in df and make config to work with one-hot
    :param config:
    :return:
    """
    ndf = encode_and_bind(config['df'], config['cat_names'], keep_old=False)
    config['df'] = ndf

    # update cat_names to [] and cont_names to all fields (except fold)
    cont_names = config['df'].select_dtypes(include=[np.number]).columns.tolist()
    cont_names = [o for o in cont_names if o not in ['Survival.time', SURVIVAL_C, config['dep_var']]]
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


@deprecation.deprecated(details='consider to use data_load_df_general. Note, difference of data_test type')
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
            .databunch(bs=config.get('bs', 64)))

    print(data_train.show_batch(rows=10))
    print(data_train)

    config.update({
        'data_train': data_train,
        'data_test': data_test_df
    })

    return config


def data_load_df_general(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    Old name: data_load_dfv2
    Make a data_train, data_test and add to config
    :param config:
    :return: new config
    """
    config['df'] = config['df'].reset_index(drop=True)
    df = config['df']

    train_valid_idx = list(df[df['fold'] != config['test_fold']].index)
    test_idx = list(df[df['fold'] == config['test_fold']].index)

    cat_names_default = df.select_dtypes(include=['object']).columns.tolist()
    cont_names_default = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_names, cont_names = config.get('cat_names', cat_names_default), config.get('cont_names', cont_names_default)

    procs_default = [FillMissing, Categorify, Normalize][:2]
    procs = config.get('procs', procs_default)

    # label_cls infer from the type of config['dep_var']
    # np.int64, np.int, int: then CategoryList
    lbl_cls = FloatList if isinstance(df.iloc[0][config['dep_var']], (float, np.float)) else CategoryList
    label_from_df_params = {'cols': config['dep_var'], 'label_cls': lbl_cls}
    if label_from_df_params['label_cls'] == FloatList:
        label_from_df_params['log'] = config['is_log']

    # careful random separated train and validation
    valid_rate = config.get('valid_pct', 0.1)
    random_seed = config.get('seed', None)
    train_idx, valid_idx = train_test_split(train_valid_idx, test_size=valid_rate, random_state=random_seed)

    # some case, we use only subset of train instead of full
    # this code will support keep valid/test unchanged when select sample for training set
    # if does not need prevent valid, consider use `prlab.medical.data_helper.DfRateKeepTrainFilter`
    # if does not need prevent test, consider use `prlab.medical.data_helper.DfRateKeepFilter`
    if config.get('train_sampling_rate', None) is not None:
        assert 0.0 <= config['train_sampling_rate'] <= 1.0
        if config.get('train_sampling_seed', None) is not None:
            np.random.seed(config['train_sampling_seed'])

        train_idx_n = np.random.choice(train_idx, len(train_idx) * config['train_sampling_rate'])
        train_idx = list(train_idx_n)

    train_valid_df = df.iloc[train_idx + valid_idx].copy().reset_index(drop=True)
    valid_idx_start = len(train_idx)

    def valid_fn(idx):
        return idx >= valid_idx_start

    # Train data bunch
    data_train = (
        TabularList.from_df(train_valid_df, path=config['path'],
                            cat_names=cat_names, cont_names=cont_names,
                            procs=procs)
            .split_by_valid_func(valid_fn)
            .label_from_df(**label_from_df_params)
            .databunch(bs=config.get('bs', 64)))

    train_test_df = df.iloc[train_idx + test_idx].copy().reset_index(drop=True)
    test_idx_start = valid_idx_start

    def test_fn(idx, **kwargs):
        # bug: (fixed) idx is not the index from the original DataFrame, this seems the index in list of items
        # then the df should be reset_index to prevent affect
        return idx >= test_idx_start

    # bug: (fixed)
    #   some cases, for cat_names, number of kinds in data_test are more than in data_train (e.g. 1,2)
    #   (mean this kind only occur in validation but not in train in split_by_rand_pct step)
    #   => embedding step later will be fail like "srcIndex < srcSelectDimSize" (at test step)
    #   ALSO, make sure the embedding order is the same as in data_train (same index order)

    # to get two sets for test only, and add to data_test later
    # should be from data_train_df to keep the categories labels order
    data_test = (
        TabularList.from_df(train_test_df, path=config['path'],
                            cat_names=cat_names, cont_names=cont_names,
                            procs=procs)
            .split_by_valid_func(test_fn)
            .label_from_df(**label_from_df_params)
            .databunch(bs=config.get('bs', 64)))

    print(data_train.show_batch(rows=10))
    print(data_train)
    print(data_test)

    config.update({
        'data_train': data_train,
        'data_test': data_test
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


class XFilter:
    """ Some wide used filter for DataFrame input.
    function form: lambda df: df
    """
    HAS_WEIGHT_FILTER = lambda df: df[df[XConst.WEIGHT_C].notnull()]
    HAS_SIZE_FILTER = lambda df: df[df[XConst.SIZE_C].notnull()]
    NO_NOTE_FILTER = lambda df: df[df[XConst.NOTE_C].isnull()] if XConst.NOTE_C in list(df.columns) else df
    ONLY_DEAD_FILTER = lambda df: df[df[XConst.DEAD_STATUS_C] == XConst.DEAD_STATUS_V]
    COMMON_WEIGHT_SIZE_FILTER = lambda df: XFilter.HAS_SIZE_FILTER(XFilter.HAS_WEIGHT_FILTER(df))


class XTransform:
    """
    Some common transform functions for DataFrame
    function form: lambda df: None (modified df itself, does not need consider output)
    """
    GENDER_TFS = lambda df: XTransform._modified(df, XConst.GENDER_C, [('male', 0), ('female', 1)])
    DAY_TO_YEAR_TFS = lambda df: XTransform._apply_fn(df, XConst.SURVIVAL_C, lambda x: x / 365)

    @staticmethod
    def _modified(df, field_name, lst):
        """
        Now, just support binary value, then lst has only 2 elements
        :param df: DataFrame
        :param field_name:
        :param lst: [(val, map_val)]
        :return: new df, replace the field_name with map_vals
        """
        assert len(lst) > 0
        f_el = lst[0]
        if not isinstance(f_el, tuple):
            lst = [(o, idx) for idx, o in enumerate(lst)]

        f_el = lst[0]
        # TODO update in future, now only 0 or 1
        df.loc[:, field_name] = np.where(df[field_name] == f_el[0], f_el[1], f_el[1] + 1)
        return df

    @staticmethod
    def _apply_fn(df, field_name, fn):
        """ apply a function (fn) to transform data in field_name """
        df[field_name] = df[field_name].transform(fn)
        return df


class SimpleCNUHPreProcessing:
    _xconst = XConst

    def __init__(self, **config):
        self.config = config

        self.selected_header = config.get('selected_header', None)
        # list of filter function apply to df lambda df: df
        self.filter_pre_list = config.get('filter_pre_list', [])
        # list of filter function apply to df lambda df: df
        self.filter_post_list = config.get('filter_post_list',
                                           [XFilter.COMMON_WEIGHT_SIZE_FILTER,
                                            XFilter.NO_NOTE_FILTER,
                                            XFilter.ONLY_DEAD_FILTER])

        # list of transform
        self.tfms = config.get('df_transform', [
            XTransform.GENDER_TFS,
            XTransform.DAY_TO_YEAR_TFS
        ])

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
        for o_filter in self.filter_pre_list:
            df = o_filter(df)

        return df

    def filter_post(self, df):
        """
        filter data frame after other steps
        :param df:
        :return:
        """

        # other filter in list
        for o_filter in self.filter_post_list:
            df = o_filter(df)

        return df

    def keep_header(self, df):
        if self.selected_header is not None:
            df = df[self.selected_header]
        return df

    def transform(self, df):

        for tfm in self.tfms:
            df = tfm(df)

        return df
