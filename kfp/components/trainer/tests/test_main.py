from src import trainer
from io import StringIO

import numpy as np
import pytest


class TestFetchDataset:
    def test_load_dataset_from_right_formatted_data(self):
        source = StringIO(
            "species_xf,culmen_length_mm_xf,culmen_depth_mm_xf,flipper_length_mm_xf,body_mass_g_xf\n"
            "0,0.2545454545454545,0.6666666666666666,0.15254237288135594,0.2916666666666667"
        )
        label_key = "species_xf"
        actual = trainer.load_dataset(source, label_key)
        assert actual.dtype.names == (
            "species_xf",
            "culmen_length_mm_xf",
            "culmen_depth_mm_xf",
            "flipper_length_mm_xf",
            "body_mass_g_xf",
        )
        assert actual["species_xf"].dtype == np.int64
        assert actual["culmen_length_mm_xf"].dtype == np.float64

    def test_fetch_dataset_fails_file_without_header(self):
        source = StringIO("0,1\n" "0,1")
        label_key = "species"
        with pytest.raises(IndexError):
            trainer.load_dataset(source, label_key)


class TestTrainModel:
    def test_train_model_from_small_dataset(self):
        source = np.array(
            [(0, 1, 2, 3, 4), (1, 2, 3, 4, 5), (0, 3, 4, 5, 6)],
            dtype=[
                ("species_xf", "i8"),
                ("culmen_length_mm_xf", "f8"),
                ("culmen_depth_mm_xf", "f8"),
                ("flipper_length_mm_xf", "f8"),
                ("body_mass_g_xf", 'f8')
            ],
        )
        model = trainer.train(source, "species_xf")
        assert model is not None
        assert model.predict([[1, 2, 3, 4]]) == [0]
