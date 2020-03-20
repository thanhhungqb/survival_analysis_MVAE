from fastai.tabular import *

from prlab.medical.cnuh_selected import cnuh_data_transform

keep_m_code_lst = ['m8041/3', 'm8070/3', 'm8140/3']


def data_load_df(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    Make a data_train, data_test and add to config
    :param config:
    :return: new config
    """

    pp = config['path'] / config['train_file']

    df = pd.read_excel(pp, sheet_name=config['sheet_name'])
    df.head()

    df = df[df['M-code(조직형)'].isin(keep_m_code_lst)].copy()
    print('data len', len(df))

    data_train_df = df[df['fold'] != config['test_fold']].copy()
    data_test_df = df[df['fold'] == config['test_fold']].copy()
    # test_data, train_data

    data_train_df = cnuh_data_transform(data_train_df)
    data_test_df = cnuh_data_transform(data_test_df)

    # cat_names = train_data.select_dtypes(include=['object']).columns.tolist()
    # cont_names = train_data.select_dtypes(include=[np.number]).columns.tolist()
    cat_names, cont_names = config['cat_names'], config['cont_names']

    procs = [FillMissing, Categorify, Normalize]

    # Test tabularlist
    test = TabularList.from_df(data_test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)

    # Train data bunch
    data_train = (
        TabularList.from_df(data_train_df, path=config['path'], cat_names=cat_names, cont_names=cont_names, procs=procs)
            .split_by_rand_pct(valid_pct=0.1, seed=42)
            .label_from_df(cols=config['dep_var'], label_cls=FloatList, log=config['is_log'])
            .add_test(test)
            .databunch())

    data_train.show_batch(rows=10)
    config.update({
        'data_train': data_train,
        'data_test': data_test_df
    })

    return config
