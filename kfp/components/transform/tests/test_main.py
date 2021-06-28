from io import StringIO

import numpy as np
import pytest
from numpy.ma.testutils import assert_equal
from src import transform


class TestFetchDataset:
    def test_fetch_dataset_from_right_formatted_data(self):
        source = StringIO(
            "species,culmen_length_mm,culmen_depth_mm,flipper_length_mm,body_mass_g\n"
            "0,0.2545454545454545,0.6666666666666666,0.15254237288135594,0.2916666666666667"
        )
        actual = transform.fetch_dataset(source)
        assert actual.dtype.names == (
            "species",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        )
        assert actual["species"].dtype == np.int64
        assert actual["culmen_length_mm"].dtype == np.float64

    def test_fetch_dataset_fails_file_without_header(self):
        source = StringIO("0,1\n" "0,1")
        with pytest.raises(IndexError):
            transform.fetch_dataset(source)


class TestTransformDataset:
    def test_transform_dataset_from_right_formated_data(self):
        source = [(0, 0.1, 0.2, 0.3, 0.4)]
        names = [
            "species",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        formats = ["i8", "f8", "f8", "f8", "f8"]
        dtypes = list(zip(names, formats))
        data = np.array(source, dtype=dtypes)

        expected_names = [name + "_xf" for name in names]
        expected_dtypes = list(zip(expected_names, formats))
        expected = np.array(source, dtype=expected_dtypes)

        actual = transform.transform(data, "_xf")
        assert_equal(actual, expected)
