"""
Author: Eckert ZHANG
Date: 2021-07-11 22:59:14
LastEditTime: 2022-03-31 20:13:42
LastEditors: Eckert ZHANG
FilePath: /pixel-nerf-portrait/src/data/__init__.py
Description: 
"""
import os
from .Facepose_WildData_static import Facepose_WildData_static
from .Facepose_WildData_ADmixexp import FP_WildData_ADmixexp


def get_split_dataset(dataset_type,
                      datadir,
                      want_split="all",
                      training=True,
                      n_view_in=3,
                      **kwargs):
    """
    Retrieved desired dataset class
    :param dataset_type dataset type name (srn|dvr|dvr_gen, etc)
    :param datadir root directory name for the dataset. For SRN/multi_obj data:
    if data is in dir/cars_train, dir/cars_test, ... then put dir/cars
    :param want_split root directory name for the dataset
    :param training set to False in eval scripts
    """
    dset_class, train_aug = None, None
    flags, train_aug_flags = {}, {}
    flags["n_view_in"] = n_view_in

    if dataset_type == "fp_admixexp":
        dset_class = FP_WildData_ADmixexp
    else:
        raise NotImplementedError("Unsupported dataset type", dataset_type)

    want_train = want_split != "val" and want_split != "test"
    want_val = want_split != "train" and want_split != "test"
    want_test = want_split != "train" and want_split != "val"

    if want_train:
        train_set = dset_class(datadir, stage="train", **flags, **kwargs)
        if train_aug is not None:
            train_set = train_aug(train_set, **train_aug_flags)

    if want_val:
        val_set = dset_class(datadir, stage="val", **flags, **kwargs)

    if want_test:
        test_set = dset_class(datadir, stage="test", **flags, **kwargs)

    if want_split == "train":
        return train_set
    elif want_split == "val":
        return val_set
    elif want_split == "test":
        return test_set
    return train_set, val_set, test_set
