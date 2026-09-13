#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import numpy as np

from src.swedish_and_foreigners import swedish_and_foreigners, main


class TestSwedishAndForeigners(unittest.TestCase):

    def test_shape(self):
        df = swedish_and_foreigners()
        self.assertEqual(
            df.shape,
            (28, 3),
            msg="swedish_and_foreigners() should return a DataFrame with "
            "shape (28, 3). Got %r." % (df.shape,),
        )

    def test_columns(self):
        df = swedish_and_foreigners()
        np.testing.assert_array_equal(
            df.columns,
            [
                "Population",
                "Share of Swedish-speakers of the population, %",
                "Share of foreign citizens of the population, %",
            ],
            err_msg="The DataFrame's columns should be exactly "
            "['Population', 'Share of Swedish-speakers of the population, "
            "%%', 'Share of foreign citizens of the population, %%'] in "
            "that order. Got %r." % (list(df.columns),),
        )

    def test_index(self):
        df = swedish_and_foreigners()
        self.assertEqual(
            df.index[0],
            "Brändö",
            msg="The first row's index should be 'Brändö'. Got "
            "%r." % (df.index[0],),
        )
        self.assertEqual(
            df.index[-1],
            "Vöyri",
            msg="The last row's index should be 'Vöyri'. Got %r." % (
                df.index[-1],
            ),
        )

    def test_content(self):
        df = swedish_and_foreigners()
        values = [452, 89.7, 10.5]
        for i in range(3):
            self.assertEqual(
                df.iloc[i, i],
                values[i],
                msg="The value on row %r, column %r should be %r. Got %r."
                % (df.index[i], df.columns[i], values[i], df.iloc[i, i]),
            )

    def test_called(self):
        with patch(
            "src.swedish_and_foreigners.swedish_and_foreigners",
            wraps=swedish_and_foreigners,
        ) as psaf:
            main()
            psaf.assert_called()


if __name__ == "__main__":
    unittest.main()
