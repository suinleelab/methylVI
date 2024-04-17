from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, _constants, fields
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.data._utils import get_anndata_attribute
from scvi.data.fields import (
    LabelsWithUnlabeledObsField,
)
from scvi.model._utils import (
    get_max_epochs_heuristic,
    parse_device_args,
)
from scvi.model.base import ArchesMixin, BaseModelClass, VAEMixin
from scvi.model.base._archesmixin import _get_loaded_data, _set_params_online_update
from scvi.model.base._utils import (
    _initialize_model,
    _validate_var_names,
)
from scvi.train import SemiSupervisedTrainingPlan, TrainRunner
from scvi.train._callbacks import SubSampleLabels
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp

from methyl_vi import METHYLVI_REGISTRY_KEYS
from methyl_vi.dataloaders import SemiSupervisedDataSplitter
from methyl_vi.module.methylanvae import METHYLANVAE

_SCANVI_LATENT_QZM = "_scanvi_latent_qzm"
_SCANVI_LATENT_QZV = "_scanvi_latent_qzv"
_SCANVI_OBSERVED_LIB_SIZE = "_scanvi_observed_lib_size"

logger = logging.getLogger(__name__)


class MethylANVI(VAEMixin, ArchesMixin, BaseModelClass):
    """Single-cell annotation using variational inference :cite:p:`Xu21`.

    Inspired from M1 + M2 model, as described in (https://arxiv.org/pdf/1406.5298.pdf).

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    linear_classifier
        If ``True``, uses a single linear layer for classification instead of a
        multi-layer perceptron.
    **model_kwargs
        Keyword args for :class:`~scvi.module.SCANVAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCANVI.setup_anndata(adata, batch_key="batch", labels_key="labels")
    >>> vae = scvi.model.SCANVI(adata, "Unknown")
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obs["pred_label"] = vae.predict()

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/scrna/harmonization`
    2. :doc:`/tutorials/notebooks/scrna/scarches_scvi_tools`
    3. :doc:`/tutorials/notebooks/scrna/seed_labeling`
    """

    _module_cls = METHYLANVAE
    _training_plan_cls = SemiSupervisedTrainingPlan

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        likelihood: Literal["betabinomial", "binomial"] = "betabinomial",
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        linear_classifier: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)
        scanvae_model_kwargs = dict(model_kwargs)

        self._set_indices_and_labels()

        # ignores unlabeled catgegory
        n_labels = self.summary_stats.n_labels - 1
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        n_batch = self.summary_stats.n_batch

        adata_manager = self.get_anndata_manager(adata)
        if type(adata) == AnnData:
            self.modalities = []
            n_input = adata.layers["cov"].shape[1]
        else:
            self.modalities = [
                x.split("_")[0] for x in adata_manager.data_registry.keys() if x.endswith("cov")
            ]
            self.num_features_per_modality = [
                adata[modality].shape[1] for modality in self.modalities
            ]
            n_input = np.sum(self.num_features_per_modality)

        self.module = self._module_cls(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            likelihood=likelihood,
            linear_classifier=linear_classifier,
            modalities=self.modalities,
            num_features_per_modality=self.num_features_per_modality,
            **scanvae_model_kwargs,
        )

        self.unsupervised_history_ = None
        self.semisupervised_history_ = None

        self._model_summary_string = (
            f"MethylanVI Model with the following params: \nunlabeled_category: "
            f"{self.unlabeled_category_}, n_hidden: {n_hidden}, n_latent: {n_latent}"
            f", n_layers: {n_layers}, dropout_rate: {dropout_rate}, dispersion: "
            f"{dispersion}, likelihood: {likelihood}"
        )
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.n_labels = n_labels

    def _set_indices_and_labels(self):
        """Set indices for labeled and unlabeled cells."""
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self.unlabeled_category_ = labels_state_registry.unlabeled_category

        labels = get_anndata_attribute(
            self.adata,
            self.adata_manager.data_registry.labels.attr_name,
            self.original_label_key,
        ).ravel()
        self._label_mapping = labels_state_registry.categorical_mapping

        # set unlabeled and labeled indices
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category_).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category_).ravel()
        self._code_to_label = dict(enumerate(self._label_mapping))

    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        soft: bool = False,
        batch_size: int | None = None,
        use_posterior_mean: bool = True,
    ) -> np.ndarray | pd.DataFrame:
        """Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
        indices
            Return probabilities for each class label.
        soft
            If True, returns per class probabilities
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        use_posterior_mean
            If ``True``, uses the mean of the posterior distribution to predict celltype
            labels. Otherwise, uses a sample from the posterior distribution - this
            means that the predictions will be stochastic.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )
        y_pred = []
        for _, tensors in enumerate(scdl):
            mc, cov = self.module._get_methylation_features(tensors)  # (n_obs, n_vars)
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]

            pred = self.module.classify(
                mc,
                cov,
                batch_index=batch,
                use_posterior_mean=use_posterior_mean,
            )
            if self.module.classifier.logits:
                pred = torch.nn.functional.softmax(pred, dim=-1)
            if not soft:
                pred = pred.argmax(dim=1)
            y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred).numpy()
        if not soft:
            predictions = []
            for p in y_pred:
                predictions.append(self._code_to_label[p])

            return np.array(predictions)
        else:
            n_labels = len(pred[0])
            pred = pd.DataFrame(
                y_pred,
                columns=self._label_mapping[:n_labels],
                index=adata.obs_names[indices],
            )
            return pred

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        n_samples_per_label: float | None = None,
        check_val_every_n_epoch: int | None = None,
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset for semisupervised training.
        n_samples_per_label
            Number of subsamples for each label class to sample per epoch. By default, there
            is no label subsampling.
        check_val_every_n_epoch
            Frequency with which metrics are computed on the data for validation set for both
            the unsupervised and semisupervised trainers. If you'd like a different frequency for
            the semisupervised trainer, set check_val_every_n_epoch in semisupervised_train_kwargs.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train,
            and test set are split in the sequential order of the data according to
            `validation_size` and `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        %(param_accelerator)s
        %(param_devices)s
        datasplitter_kwargs
            Additional keyword arguments passed into
            :class:`~scvi.dataloaders.SemiSupervisedDataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.SemiSupervisedTrainingPlan`. Keyword
            arguments passed to `train()` will overwrite values present in `plan_kwargs`,
            when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))

        logger.info(f"Training for {max_epochs} epochs.")

        plan_kwargs = {} if plan_kwargs is None else plan_kwargs
        datasplitter_kwargs = datasplitter_kwargs or {}

        # if we have labeled cells, we want to subsample labels each epoch
        sampler_callback = [SubSampleLabels()] if len(self._labeled_indices) != 0 else []

        data_splitter = SemiSupervisedDataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            n_samples_per_label=n_samples_per_label,
            batch_size=batch_size,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, self.n_labels, **plan_kwargs)

        if "callbacks" in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] + [sampler_callback]
        else:
            trainer_kwargs["callbacks"] = sampler_callback

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **trainer_kwargs,
        )
        return runner()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        mdata: MuData,
        mc_layer: str,
        cov_layer: str,
        labels_key: str,
        unlabeled_category: str,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        methylation_modalities: dict[str, str] | None = None,
        covariate_modalities=None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_labels_key)s
        %(param_unlabeled_category)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        if covariate_modalities is None:
            covariate_modalities = {}
        setup_method_args = MethylANVI._get_setup_method_args(**locals())

        if methylation_modalities is None:
            raise ValueError("Methylation modalities cannot be None.")

        covariate_modalities_ = cls._create_modalities_attr_dict(
            covariate_modalities, setup_method_args
        )

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=covariate_modalities_.batch_key,
        )

        mc_fields = []
        cov_fields = []

        for mod in methylation_modalities:
            mc_fields.append(
                fields.MuDataLayerField(
                    f"{mod}_{METHYLVI_REGISTRY_KEYS.MC_KEY}",
                    mc_layer,
                    mod_key=methylation_modalities[mod],
                    is_count_data=True,
                    mod_required=True,
                )
            )

            cov_fields.append(
                fields.MuDataLayerField(
                    f"{mod}_{METHYLVI_REGISTRY_KEYS.COV_KEY}",
                    cov_layer,
                    mod_key=methylation_modalities[mod],
                    is_count_data=True,
                    mod_required=True,
                )
            )

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=covariate_modalities_.batch_key,
        )

        mudata_fields = (
            mc_fields
            + cov_fields
            + [
                batch_field,
                LabelsWithUnlabeledObsField(
                    REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category
                ),
                fields.MuDataCategoricalJointObsField(
                    REGISTRY_KEYS.CAT_COVS_KEY,
                    categorical_covariate_keys,
                    mod_key=covariate_modalities_.categorical_covariate_keys,
                ),
                fields.MuDataNumericalJointObsField(
                    REGISTRY_KEYS.CONT_COVS_KEY,
                    continuous_covariate_keys,
                    mod_key=covariate_modalities_.continuous_covariate_keys,
                ),
            ]
        )
        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        mc_layer: str,
        cov_layer: str,
        labels_key: str,
        unlabeled_category: str,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        methylation_modalities: dict[str, str] | None = None,
        covariate_modalities=None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_labels_key)s
        %(param_unlabeled_category)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        if covariate_modalities is None:
            covariate_modalities = {}
        setup_method_args = MethylANVI._get_setup_method_args(**locals())

        if methylation_modalities is None:
            raise ValueError("Methylation modalities cannot be None.")

        covariate_modalities_ = cls._create_modalities_attr_dict(
            covariate_modalities, setup_method_args
        )

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=covariate_modalities_.batch_key,
        )

        mc_fields = []
        cov_fields = []

        for mod in methylation_modalities:
            mc_fields.append(
                fields.MuDataLayerField(
                    f"{mod}_{METHYLVI_REGISTRY_KEYS.MC_KEY}",
                    mc_layer,
                    mod_key=methylation_modalities[mod],
                    is_count_data=True,
                    mod_required=True,
                )
            )

            cov_fields.append(
                fields.MuDataLayerField(
                    f"{mod}_{METHYLVI_REGISTRY_KEYS.COV_KEY}",
                    cov_layer,
                    mod_key=methylation_modalities[mod],
                    is_count_data=True,
                    mod_required=True,
                )
            )

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=covariate_modalities_.batch_key,
        )

        mudata_fields = (
            mc_fields
            + cov_fields
            + [
                batch_field,
                LabelsWithUnlabeledObsField(
                    REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category
                ),
                fields.MuDataCategoricalJointObsField(
                    REGISTRY_KEYS.CAT_COVS_KEY,
                    categorical_covariate_keys,
                    mod_key=covariate_modalities_.categorical_covariate_keys,
                ),
                fields.MuDataNumericalJointObsField(
                    REGISTRY_KEYS.CONT_COVS_KEY,
                    continuous_covariate_keys,
                    mod_key=covariate_modalities_.continuous_covariate_keys,
                ),
            ]
        )
        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    def load_query_mdata(
        cls,
        mdata: MuData,
        reference_model: str | BaseModelClass,
        inplace_subset_query_vars: bool = False,
        accelerator: str = "auto",
        device: int | str = "auto",
        unfrozen: bool = False,
        freeze_dropout: bool = False,
        freeze_expression: bool = True,
        freeze_decoder_first_layer: bool = True,
        freeze_batchnorm_encoder: bool = True,
        freeze_batchnorm_decoder: bool = False,
        freeze_classifier: bool = True,
    ):
        """Online update of a reference model with scArches algorithm :cite:p:`Lotfollahi21`.

        Parameters
        ----------
        adata
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
        reference_model
            Either an already instantiated model of the same class, or a path to
            saved outputs for reference model.
        inplace_subset_query_vars
            Whether to subset and rearrange query vars inplace based on vars used to
            train reference model.
        %(param_accelerator)s
        %(param_device)s
        unfrozen
            Override all other freeze options for a fully unfrozen model
        freeze_dropout
            Whether to freeze dropout during training
        freeze_expression
            Freeze neurons corersponding to expression in first layer
        freeze_decoder_first_layer
            Freeze neurons corersponding to first layer in decoder
        freeze_batchnorm_encoder
            Whether to freeze batchnorm weight and bias during training for encoder
        freeze_batchnorm_decoder
            Whether to freeze batchnorm weight and bias during training for decoder
        freeze_classifier
            Whether to freeze classifier completely. Only applies to `SCANVI`.
        """
        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        attr_dict, var_names, load_state_dict = _get_loaded_data(reference_model, device=device)

        if inplace_subset_query_vars:
            logger.debug("Subsetting query vars to reference vars.")
            mdata._inplace_subset_var(var_names)
        _validate_var_names(mdata, var_names)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError(
                "Saved model does not contain original setup inputs. "
                "Cannot load the original setup."
            )

        cls.setup_anndata(
            mdata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, mdata, attr_dict)
        adata_manager = model.get_anndata_manager(mdata, required=True)

        if REGISTRY_KEYS.CAT_COVS_KEY in adata_manager.data_registry:
            raise NotImplementedError(
                "scArches currently does not support models with extra categorical covariates."
            )

        version_split = adata_manager.registry[_constants._SCVI_VERSION_KEY].split(".")
        if int(version_split[1]) < 8 and int(version_split[0]) == 0:
            warnings.warn(
                "Query integration should be performed using models trained with "
                "version >= 0.8",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        model.to_device(device)

        # model tweaking
        new_state_dict = model.module.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                load_state_dict[key] = fixed_ten

        model.module.load_state_dict(load_state_dict)
        model.module.eval()

        _set_params_online_update(
            model.module,
            unfrozen=unfrozen,
            freeze_decoder_first_layer=freeze_decoder_first_layer,
            freeze_batchnorm_encoder=freeze_batchnorm_encoder,
            freeze_batchnorm_decoder=freeze_batchnorm_decoder,
            freeze_dropout=freeze_dropout,
            freeze_expression=freeze_expression,
            freeze_classifier=freeze_classifier,
        )
        model.is_trained_ = False

        return model
