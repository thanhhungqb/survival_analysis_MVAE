from pathlib import Path

import numpy as np
from scipy.stats import norm
from torch.utils.data import Dataset

from prlab.common.utils import convert_to_obj_or_fn, CategoricalEncoderPandas


class SliceDataset(Dataset):
    """
    Implement Slice dataset of body including many slices, that make 3D of human body
    Data is store in numpy and meta in df
    Work with pytorch
    """

    def __init__(self, df, path, transform=None, map_name_fn=None, check_file=True, **config):
        """
        data: extracted npy data with shape (slice_count x H_size x W_size
        NOTE: df should be .reset_index(drop=True) to make [0, len) otherwise access by index will be wrong
        """
        super(SliceDataset, self).__init__()

        self.df = df
        self.path = Path(path)
        self.transform = transform
        self.map_name_fn = convert_to_obj_or_fn(map_name_fn) if map_name_fn is not None else (lambda pid: f"{pid}.npy")
        if check_file:
            file_ok = [(idx, self.path / self.map_name_fn(self.df['pid'][idx])) for idx in range(len(self.df))]
            file_ok_idx = [idx for idx, o in file_ok if o.is_file()]

            logger = config.get('train_logger', None)
            if logger is not None:
                s = set(file_ok_idx)
                not_found = [o for o in range(len(self.df)) if o not in s]
                not_found_files = [self.path / self.map_name_fn(self.df['pid'][idx]) for idx in not_found]
                msg = "\n".join([str(o) for o in not_found_files])
                logger.warning(f"several file not found to load: \n{msg}")

            self.df = self.df.loc[file_ok_idx].reset_index(drop=True)

    def __getitem__(self, index):
        data_path = self.path / self.map_name_fn(self.df['pid'][index])
        slices = np.load(data_path)  # n_slice x H X W

        if self.transform:
            slices = self.transform(slices)

        slices = np.expand_dims(slices, axis=0).astype(np.float32)
        return slices, np.array(self.df['Survival.time'][index]).astype(np.float32)

    def __len__(self):
        return len(self.df)


class ClinicalDataset(Dataset):
    """
    Implement clinical dataset data loader.
    NOTE: df should be .reset_index(drop=True) to make [0, len) otherwise access by index will be wrong
    The index should be shared between train/test/valid, none available should return 0, cat_encoder,
    then cat_encoder should be make outside and pass here, or make here (if train) and get to pass to valid/test
    support multi-value of y from multi-column, but they should be same type (cont or cat)
    Work with pytorch.
    """

    def __init__(self, df, cat_names=None, cont_names=None, y_names=None, cat_encoder=None,
                 label_fn=None,
                 float_type=np.float32, int_type=np.int64,
                 **config):
        super(ClinicalDataset, self).__init__()

        self.df = df.reset_index(drop=True)

        if cat_names is None: cat_names = sorted(df.select_dtypes(include=['object']).columns.tolist())
        if cont_names is None: cont_names = sorted(df.select_dtypes(include=[np.number]).columns.tolist())

        self.cat_names, self.cont_names = cat_names, cont_names
        self.y_names = y_names if isinstance(y_names, list) else [y_names]

        y_types = [df[o].dtypes in [np.float, np.float32, np.float64] for o in self.y_names]
        self.y_type = float_type if np.any(y_types) else int_type

        if cat_encoder is None:
            cat_encoder = CategoricalEncoderPandas(cat_names=cat_names)
            cat_encoder.fit_df(df=df)
        self.cat_encoder = cat_encoder

        # type to use, default is float32 and int64, but maybe change in future
        self.float_type, self.int_type = float_type, int_type

        self.label_fn = convert_to_obj_or_fn(label_fn) if label_fn is not None else (lambda x, **kw: x)

    def __getitem__(self, index):
        cat_values = [self.df[o][index] for o in self.cat_names]
        cat_values = [self.cat_encoder.transform(name, [str(value)])[0]
                      for name, value in zip(self.cat_names, cat_values)]

        cont_values = [self.df[o][index] for o in self.cont_names]

        y_values = [self.df[o][index] for o in self.y_names]
        y_values = np.array(y_values, self.y_type)
        y_values = self.label_fn(y_values)

        cat_values = np.array(cat_values, dtype=self.int_type).reshape(-1)
        cont_values = np.array(cont_values, dtype=self.float_type)

        return (cat_values, cont_values), y_values

    def __len__(self):
        return len(self.df)


class SurvivalRegClsLabelFn:
    """
    Join regression and binary classification for survival time.
    x should be [single_float], return [float] + [int(0/1)] (float[2])
    """

    def __init__(self, sep_value, scale=1., dtype=np.float32, **kwargs):
        """
        :param sep_value: for binary classes separated, before scale
        :param scale: e.g. 365 to convert from days to years, 1/365 if reverse
        :param dtype: default is float32, but maybe down to float16 later
        :param kwargs:
        """
        self.scale = scale  # x = x / scale
        self.sep_value = sep_value
        self.dtype = dtype

    def __call__(self, x, *args, **kwargs):
        cls = 0 if x < self.sep_value else 1
        x = x / self.scale
        return np.array([x, cls], dtype=self.dtype).reshape(-1)


class SurvivalRegClsProbLabelFn(SurvivalRegClsLabelFn):
    """
    Join regression and binary classification for survival time.
    x should be [single_float], return [float] + [prob_0, prob_1] (float[3]).
    NOTE: We assume the survival time for each patient is normal distribution and std linear related to x,
    std = x * std_rate, e.g. 10 years of survival time with std 1 year.
    """

    def __init__(self, sep_value, std_rate=0.1, scale=1., dtype=np.float32, **kwargs):
        """
        :param sep_value: for binary classes separated, before scale
        :param std_rate: for calculate std based on x
        :param scale: e.g. 365 to convert from days to years, 1/365 if reverse
        :param kwargs:
        """
        super(SurvivalRegClsProbLabelFn, self).__init__(sep_value=sep_value, scale=scale, dtype=dtype, **kwargs)
        self.std_rate = std_rate

    def __call__(self, x, *args, **kwargs):
        dis = norm(x, x * self.std_rate)
        p_lh = dis.cdf(self.sep_value)
        p_lh = [p_lh, 1 - p_lh]
        x = x / self.scale
        return np.array([x, *p_lh], dtype=self.dtype).reshape(-1)
