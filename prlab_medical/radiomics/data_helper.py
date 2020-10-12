import scipy
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from outside.medical_image_pre_aug import elastic_transform_3d
from prlab.data_process.augmentation import rand_crop_near_center
from prlab.torch.functions import TransformsWrapFn
from prlab_medical.radiomics.radiology import SliceDataset


def patient_data_loader(**config):
    """
    Pipe for data loader.
    Given info:
        path: base path of files (npy)
        df: loaded
        tfms/tfms_test
    :param config:
    :return: config with updated
    """
    train_df, test_df = config['train_df'], config['test_df']
    valid_df = config.get('valid_df', None)

    train_tfms = config.get('tfms', [])
    test_tfms = config.get('tfms_test', train_tfms)  # test aug
    train_tfms, test_tfms = transforms.Compose(train_tfms), transforms.Compose(test_tfms)

    train_dataset = SliceDataset(transform=train_tfms, **{**config, 'df': train_df})
    test_dataset = SliceDataset(transform=test_tfms, **{**config, 'df': test_df})
    valid_dataset = SliceDataset(transform=test_tfms, **{**config, 'df': valid_df}) if valid_df is not None else None

    kk = ['batch_size', 'shuffle', 'sampler', 'batch_sampler', 'num_workers', 'collate_fn',
          'pin_memory', 'drop_last', 'timeout', 'worker_init_fn', 'multiprocessing_context']
    kw = {k: config[k] for k in kk if k in config.keys()}
    kw = {**kw, 'batch_size': config.get('bs', 8)}

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **kw)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, **kw)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, **kw) if valid_dataset is not None else None

    config.update({
        'data_train': train_loader,
        'data_test': test_loader,
        'data_valid': valid_loader
    })
    return config


def tfms_general_rad(**config):
    """
    Using with pytorch
    :param config:
    :return:
    """
    to_tensor = lambda slices: torch.tensor(slices)

    tfms = [
        # TransformsWrapFn(random_rotate_xy, angle=config.get('rotate_angle', [-30, 30])),
        TransformsWrapFn(rand_crop_near_center,
                         crop_size=config.get('crop_size', (224, 224)),
                         d=config.get('d_crop', [20, 20])),
        TransformsWrapFn(scipy.ndimage.interpolation.zoom,
                         zoom=config.get('zoom', [1, 0.5, 0.5]), order=config.get('zoom_order', 0)),
        TransformsWrapFn(elastic_transform_3d),
        to_tensor,
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]
    tfms_test = tfms

    config.update({'tfms': tfms, 'tfms_test': tfms_test})
    return config
