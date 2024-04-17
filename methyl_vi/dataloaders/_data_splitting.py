from typing import Optional

import lightning.pytorch as pl
import numpy as np
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from scvi.dataloaders._data_splitting import validate_data_split
from scvi.dataloaders._semi_dataloader import SemiSupervisedDataLoader


class SemiSupervisedDataSplitter(pl.LightningDataModule):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.
    The ratio between labeled and unlabeled data in adata will be preserved
    in the train/test/val sets.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    shuffle_set_split
        Whether to shuffle indices before splitting. If `False`, the val, train, and test set
        are split in the sequential order of the data according to `validation_size` and
        `train_size` percentages.
    n_samples_per_label
        Number of subsamples for each label class to sample per epoch
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~scvi.data.AnnDataLoader`.
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata, labels_key="labels")
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> unknown_label = 'label_0'
    >>> splitter = SemiSupervisedDataSplitter(adata, unknown_label)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        n_samples_per_label: Optional[int] = None,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.data_loader_kwargs = kwargs
        self.n_samples_per_label = n_samples_per_label

        labels_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name,
            labels_state_registry.original_key,
        ).ravel()
        self.unlabeled_category = labels_state_registry.unlabeled_category
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category).ravel()

        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_labeled_idx = len(self._labeled_indices)
        n_unlabeled_idx = len(self._unlabeled_indices)

        if n_labeled_idx != 0:
            n_labeled_train, n_labeled_val = validate_data_split(
                n_labeled_idx, self.train_size, self.validation_size
            )

            labeled_permutation = self._labeled_indices
            if self.shuffle_set_split:
                rs = np.random.RandomState(seed=settings.seed)
                labeled_permutation = rs.choice(
                    self._labeled_indices, len(self._labeled_indices), replace=False
                )

            labeled_idx_val = labeled_permutation[:n_labeled_val]
            labeled_idx_train = labeled_permutation[
                n_labeled_val : (n_labeled_val + n_labeled_train)
            ]
            labeled_idx_test = labeled_permutation[(n_labeled_val + n_labeled_train) :]
        else:
            labeled_idx_test = []
            labeled_idx_train = []
            labeled_idx_val = []

        if n_unlabeled_idx != 0:
            n_unlabeled_train, n_unlabeled_val = validate_data_split(
                n_unlabeled_idx, self.train_size, self.validation_size
            )

            unlabeled_permutation = self._unlabeled_indices
            if self.shuffle_set_split:
                rs = np.random.RandomState(seed=settings.seed)
                unlabeled_permutation = rs.choice(
                    self._unlabeled_indices, len(self._unlabeled_indices), replace=False
                )

            unlabeled_idx_val = unlabeled_permutation[:n_unlabeled_val]
            unlabeled_idx_train = unlabeled_permutation[
                n_unlabeled_val : (n_unlabeled_val + n_unlabeled_train)
            ]
            unlabeled_idx_test = unlabeled_permutation[(n_unlabeled_val + n_unlabeled_train) :]
        else:
            unlabeled_idx_train = []
            unlabeled_idx_val = []
            unlabeled_idx_test = []

        indices_train = np.concatenate((labeled_idx_train, unlabeled_idx_train))
        indices_val = np.concatenate((labeled_idx_val, unlabeled_idx_val))
        indices_test = np.concatenate((labeled_idx_test, unlabeled_idx_test))

        self.train_idx = indices_train.astype(int)
        self.val_idx = indices_val.astype(int)
        self.test_idx = indices_test.astype(int)

        if len(self._labeled_indices) != 0:
            self.data_loader_class = SemiSupervisedDataLoader
            dl_kwargs = {
                "n_samples_per_label": self.n_samples_per_label,
            }
        else:
            self.data_loader_class = AnnDataLoader
            dl_kwargs = {}

        self.data_loader_kwargs.update(dl_kwargs)

    def train_dataloader(self):
        """Create the train data loader."""
        return self.data_loader_class(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create the validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create the test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
