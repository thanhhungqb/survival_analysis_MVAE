def dlbcl_data_to_num(**config):
    """
    Convert some columns that mix of numeric and str, using numeric if possible
    e.g. 2 and '2, 1 vs '1
    :param config:
    :return:
    """
    df = config['df']

    num_map = \
        ['Bsymptom', 'LDH', 'NO of extranodal', 'stage', 'BM involvement', 'Bcl-2', 'beta2-microglobulin', 'IFRT',
         'relapse유무', 'PFS(days)', 'PFS(months)', '사망유무',
         'OS(days)', 'OS(months)'
         ] + ['age_stat', 'sex', 'Performance', 'Performance_stat', 'LDH_stat', 'Extranodal_stat',
              'stagescore',
              'spleen involvement',
              'IPI score', 'IPIrisk', 'R-IPI',
              'Bulky',
              'Deauville score', 'total cycle', 'PETresponse', 'final response',
              ]
    if config.get('num_map', None) is not None:
        num_map = config['num_map']

    def to_num(s):
        if isinstance(s, (int, float)):
            return s
        if isinstance(s, str):
            if s.isnumeric():
                if '.' in s: return float(s)
                return int(s)
            else:
                return s
        return s

    for name in num_map:
        tmp = [to_num(o) for o in df[name]]
        df[name] = tmp

    config['df'] = df
    return config


def dlbcl_data_norm(**config):
    df = config['df']
    # need norm (convert): LDHstat, Extranodalstat, BM involvement, Bulky, final response, IFRT, relapse유무, 사망유무,
    # age2, Bsymptom
    # Permance: 0-3 => 1-4 (+1 check)
    # not need: Bcl-2, beta2-microglobulin (float), PETresponse (final?)
    map_other = {
        'final response': {1: 'CR', 2: 'PR', 3: 'SD', 4: 'PD', 5: 'NM'},
        '사망유무': {2: 0},  # 2 sinh ton => 0
        'age2': {2: 0},  # check
        'Bsymptom': {2: 0},  # check
        'LDHstat': {2: 0},  # check
        'BM involvement': {2: 0},  # check
        'relapse유무': {2: 0},  # check
        'Extranodalstat': {2: 0},
        'Bulky': {2: 0},
        'IFRT': {2: 0},
    }

    def map_field_values(lst, mapping={}):
        out = []
        for o in lst:
            out.append(mapping[o] if o in mapping.keys() else o)
        return out

    for name in map_other.keys():
        df[name] = map_field_values(df[name], map_other[name])

    # 'Permance': {0: 1, 1: 2, 2: 3, 3: 4} need a special convert base on University
    map_permance = {0: 1, 1: 2, 2: 3, 3: 4}  # check
    new_permance = []
    for u, p in zip(df['University'], df['Permance']):
        if u == 2:
            new_permance.append(p)
        else:
            new_permance.append(map_permance[p])
    df['Permance'] = new_permance

    # df['fold'] = df['University']
    config['test_fold'] = 2  # 2 for JBUH

    df.to_excel(config['cp'] / 'all.xlsx', index=False)

    return config
