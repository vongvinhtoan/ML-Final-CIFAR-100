import dataset
from configs import settings


def test_import():
    assert settings.DATASET_PATH.exists(), f"The dataset path must exists at {settings.DATASET_PATH}"


def test_dataset_shape():
    assert len(dataset.cifar100_train) == 50_000
    assert len(dataset.cifar100_val) == 10_000
    assert dataset.cifar100_train[0][0].shape == (3, 32, 32)
    assert dataset.cifar100_val[0][0].shape == (3, 32, 32)
