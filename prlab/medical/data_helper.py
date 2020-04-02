from fastai.tabular import *

from prlab.gutils import encode_and_bind
from prlab.medical.cnuh_selected import cnuh_data_transform, selected_header_en, TNM_CODE_C, M_CODE_C

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
