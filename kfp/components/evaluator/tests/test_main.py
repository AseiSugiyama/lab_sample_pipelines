from io import StringIO

import numpy as np
import pytest
from src import evaluator


class TestLoadDataset:
    def test_load_dataset_from_right_formatted_data(self):
        source = StringIO(
            "species_xf,culmen_length_mm_xf,culmen_depth_mm_xf,flipper_length_mm_xf,body_mass_g_xf\n"
            "0,0.2545454545454545,0.6666666666666666,0.15254237288135594,0.2916666666666667"
        )
        label_key = "species_xf"
        actual = evaluator.load_dataset(source, label_key)
        assert actual.dtype.names == (
            "species_xf",
            "culmen_length_mm_xf",
            "culmen_depth_mm_xf",
            "flipper_length_mm_xf",
            "body_mass_g_xf",
        )
        assert actual["species_xf"].dtype == np.int64
        assert actual["culmen_length_mm_xf"].dtype == np.float64

    def test_load_dataset_fails_file_without_header(self):
        source = StringIO("0,1\n" "0,1")
        label_key = "species"
        with pytest.raises(IndexError):
            evaluator.load_dataset(source, label_key)

