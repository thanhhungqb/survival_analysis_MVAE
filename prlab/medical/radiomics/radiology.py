from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from prlab.gutils import convert_to_obj_or_fn


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
        return slices, self.df['Survival.time'][index]

    def __len__(self):
        return len(self.df)
