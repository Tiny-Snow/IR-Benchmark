# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Dataset
# Description:
#   This module includes Dataset used in ItemRec.
#   All datasets should be inherited from IRDataset, the standard
#   and base dataset class for ItemRec.
# -------------------------------------------------------------------

# import modules ----------------------------------------------------
from typing import (
    Any, 
    Optional,
    List,
    Tuple,
    Set,
    Dict,
    Callable,
)
import os
import csv
import random

# public functions --------------------------------------------------
__all__ = [
    'IRDataset',
]

# ItemRec Base Dataset ----------------------------------------------
class IRDataset:
    r"""
    ## Class
    The standard and base dataset class for ItemRec.

    We provide the standard `IRDataset` for ItemRec. You may inherit
    it and add the necessary methods for your own dataset.
    Note that the `IRDataset` is designed for item recommendation,
    and it is different from the standard `torch.utils.data.Dataset`.
    Specifically, the `IRDataset` has no `__getitem__` method. 

    The `IRDataset` reads the user-item interactions from the specified
    files, each dataset should at least include the following files:
    - `train.tsv`: the user-item interactions for training.
    - `test.tsv`: the user-item interactions for testing.
    The `.tsv` file is a tab-separated values file (with header), 
    each line is (user_id, item_id) pair, where id is an integer.

    IR Benchmark is a research-oriented benchmark for item recommendation.
    Thus, all datasets must be split into train and test sets before
    wrapped into dataset. However, the train and test sets should not be 
    split into two datasets. Instead, you can consider the test set as 
    the `mask` of the original user-item interaction matrix, and the
    remaining part as the train set. You should make sure the users and
    items in the test set are already appeared in the train set.

    NOTE: We randomly split the training set into train and valid set 
    with ratio 9:1. To ensure the fairness of the comparison, you should
    only use the test set for final evaluation, and the valid set for
    hyper-parameter tuning. If you do not need the valid set, you can
    set the `no_valid` flag to `True` when initializing the `IRDataset`.

    The `IRDataset` provides the following key properties:
    - user_size: the number of users in the dataset. All users are [0, user_size).
    - item_size: the number of items in the dataset. All items are [0, item_size).
    - train_size: the number of interactions in the training set.
    - valid_size: the number of interactions in the validation set.
    - test_size: the number of interactions in the testing set.
    - train_interactions: the user-item interactions for training, i.e. [(user_id, item_id)]
    - valid_interactions: the user-item interactions for validation, i.e. [(user_id, item_id)]
    - test_interactions: the user-item interactions for testing, i.e. [(user_id, item_id)]
    - train_dict: the user-item dict for the training set, i.e. (user_id: [item_ids])
        The item ids are sorted in ascending order for convenience of sampling negative items.
    - valid_dict: the user-item dict for the validation set, i.e. (user_id: [item_ids])
        The item ids are sorted in ascending order.
    - test_dict: the user-item dict for the testing set, i.e. (user_id: [item_ids])
        The item ids are sorted in ascending order.
    - resplit: re-split the training and validation sets, especially for cross-validation.

    The `IRDataset` also provides the following key methods:
    - sample_negative: sample negative items for a user randomly.

    """
    def __init__(self, data_dir: str, no_valid: Optional[bool] = False) -> None:
        r"""
        ## Function
        Initialize the `IRDataset`.

        ## Arguments
        data_dir: str
            The directory of the dataset.
        no_valid: Optional[bool]
            Whether to use the valid set. Default is `False` (use valid set).
        """
        super(IRDataset, self).__init__()
        self.data_dir = data_dir
        train_file = os.path.join(data_dir, 'train.tsv')
        test_file = os.path.join(data_dir, 'test.tsv')
        # interactions: (user_id, item_id)
        self._train_interactions = self._read_interactions(train_file)
        self._test_interactions = self._read_interactions(test_file)
        # all users and items (set), check continuous id
        users = set([user_id for user_id, _ in self._train_interactions]
            + [user_id for user_id, _ in self._test_interactions])
        items = set([item_id for _, item_id in self._train_interactions]
            + [item_id for _, item_id in self._test_interactions])
        # if need, re-map user and item ids to make them continuous
        if len(users) != max(users) + 1 or len(items) != max(items) + 1:
            user_map = {user_id: i for i, user_id in enumerate(users)}
            item_map = {item_id: i for i, item_id in enumerate(items)}
            self._train_interactions = [(user_map[user_id], item_map[item_id])
                for user_id, item_id in self._train_interactions]
            self._test_interactions = [(user_map[user_id], item_map[item_id])
                for user_id, item_id in self._test_interactions]
            users = set([user_id for user_id, _ in self._train_interactions]
                + [user_id for user_id, _ in self._test_interactions])
            items = set([item_id for _, item_id in self._train_interactions]
                + [item_id for _, item_id in self._test_interactions])
        self._user_size = len(users)
        self._item_size = len(items)
        # user-items dict (in fact a List[List[int]])
        self._train_dict = self._build_dict(self._train_interactions)
        self._test_dict = self._build_dict(self._test_interactions)
        self.no_valid = no_valid
        if not no_valid:    # split train set into train & valid set with ratio 9:1
            self._train_dict, self._valid_dict = self._split_train_valid(self._train_dict)
            self._train_interactions = [(u, i) for u, items in enumerate(self._train_dict) for i in items]
            self._valid_interactions = [(u, i) for u, items in enumerate(self._valid_dict) for i in items]
        else:               # no valid set, test set = valid set
            self._valid_dict = self._test_dict
            self._valid_interactions = self._test_interactions

    def _read_interactions(self, file_path: str) -> List[Tuple[int, int]]:
        r"""
        ## Function
        Read user-item interactions from tsv file.

        ## Arguments
        file_path: str
            The tsv file path of the user-item interactions.

        ## Returns
        interactions: List[Tuple[int, int]]
            The user-item interactions.
        """
        interactions = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                user_id, item_id = int(row[0]), int(row[1])
                interactions.append((user_id, item_id))
        return interactions
        
    def _build_dict(self, interactions: List[Tuple[int, int]]) -> List[List[int]]:
        r"""
        ## Function
        Build user-item dict from interactions.
        Specifically, the item ids are sorted in ascending order.

        ## Arguments
        interactions: List[Tuple[int, int]]
            The user-item interactions.

        ## Returns
        dict: List[List[int]]
            The user-item dict, dict[user_id] = [item_ids].
        """
        ui_dict = [[] for _ in range(self._user_size)]
        for user_id, item_id in interactions:
            ui_dict[user_id].append(item_id)
        for i in range(self._user_size):
            ui_dict[i].sort()
        return ui_dict
    
    def _split_train_valid(self, train_dict: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        r"""
        ## Function
        Split the training set into train and valid set with ratio 9:1.

        ## Arguments
        train_dict: List[List[int]]
            The user-item dict for the training set.

        ## Returns
        train_dict: List[List[int]]
            The user-item dict for the training set.
        valid_dict: List[List[int]]
            The user-item dict for the valid set.
        """
        valid_dict = [[] for _ in range(self._user_size)]
        for user_id, item_ids in enumerate(train_dict):
            valid_dict[user_id] = random.sample(item_ids, max(int(len(item_ids) * 0.1), 1))
            train_dict[user_id] = list(set(item_ids) - set(valid_dict[user_id]))
            valid_dict[user_id].sort()
            train_dict[user_id].sort()
        return train_dict, valid_dict
    
    @property
    def user_size(self) -> int:
        r"""
        ## Property
        The number of users in the dataset.
        """
        return self._user_size
    
    @property
    def item_size(self) -> int:
        r"""
        ## Property
        The number of items in the dataset.
        """
        return self._item_size
    
    @property
    def train_size(self) -> int:
        r"""
        ## Property
        The number of interactions in the training set.
        """
        return len(self._train_interactions)
    
    @property
    def valid_size(self) -> int:
        r"""
        ## Property
        The number of interactions in the validation set.
        """
        return len(self._valid_interactions)

    @property
    def test_size(self) -> int:
        r"""
        ## Property
        The number of interactions in the testing set.
        """
        return len(self._test_interactions)
    
    @property
    def train_interactions(self) -> List[Tuple[int, int]]:
        r"""
        ## Property
        The user-item interactions for training.
        """
        return self._train_interactions

    @property
    def valid_interactions(self) -> List[Tuple[int, int]]:
        r"""
        ## Property
        The user-item interactions for validation.
        """
        return self._valid_interactions

    @property
    def test_interactions(self) -> List[Tuple[int, int]]:
        r"""
        ## Property
        The user-item interactions for testing.
        """
        return self._test_interactions
    
    @property
    def train_dict(self) -> List[List[int]]:
        r"""
        ## Property
        The user-item dict for the training set.
        """
        return self._train_dict
    
    @property
    def valid_dict(self) -> List[List[int]]:
        r"""
        ## Property
        The user-item dict for the validation set.
        """
        return self._valid_dict

    @property
    def test_dict(self) -> List[List[int]]:
        r"""
        ## Property
        The user-item dict for the testing set.
        """
        return self._test_dict

    def __len__(self) -> int:
        r"""
        ## Function
        Return the total number of interactions in the dataset.
        """
        if self.no_valid:
            return self.train_size + self.test_size
        return self.train_size + self.valid_size + self.test_size

