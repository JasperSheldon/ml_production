import logging

import pytest
from sklearn.model_selection import train_test_split

# from classification_model.config.core import config
from classification_model.processing.data_manager import _load_raw_dataset

logger=logging.getLogger(__name__)


@pytest.fixture
def sample_input_data():
    data = _load_raw_dataset(file_name="titanic.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        data, 
        data["survived"],
        test_size=0.1,
        random_state=0,
    )

    return X_test
