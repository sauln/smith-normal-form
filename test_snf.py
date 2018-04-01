import numpy as np
from snf import MatrixOperations

class TestSNF():
    def test_row_flip(self):

        m = np.reshape(range(9), (3,3))
        ops = MatrixOperations(10000)
        mat = ops._flip_rows(m, 1, 2)

        assert np.array_equal(mat, [[0,1,2],[6,7,8],[3,4,5]])
        assert m is mat

    def test_flip_same_as_x_id(self):
        ops = MatrixOperations(10000)
        eye = np.identity(3)
        eyeji = ops._flip_rows(eye, 1, 2)

        m = np.reshape(range(9), (3,3))
        mcopy = np.copy(m)

        # flipped
        mat = ops._flip_rows(m, 1, 2)

        # should be the same as left multiply by the eyeji
        t = np.dot(eyeji, mcopy)

        assert np.array_equal(mat, t)

    def test_add_row(self):
        m = np.reshape(range(9), (3,3))
        ops = MatrixOperations(100000)
        mat = ops._add_row(m, 1, 2)

        assert np.array_equal(mat, [[0,1,2],[9,11,13],[6,7,8]])
        assert m is mat

    def test_add_row_mod_2(self):
        m = np.reshape(range(9), (3,3))
        ops = MatrixOperations(2)
        mat = ops._add_row(m, 1, 2)

        assert np.array_equal(mat, [[0,1,2],[1,1,1],[6,7,8]])
        assert m is mat

        mat = ops._add_row(mat, 0, 2)

        assert np.array_equal(mat, [[0,0,0],[1,1,1],[6,7,8]])
        assert m is mat

    def test_add_col(self):
        m = np.reshape(range(9), (3,3))
        ops = MatrixOperations(100000)
        mat = ops._add_col(m, 1, 2)

        assert np.array_equal(mat, [[0,3,2],[3,9,5],[6,15,8]])
        assert m is mat

    def test_add_col_mod_2(self):
        m = np.reshape(range(9), (3,3))
        ops = MatrixOperations(2)
        mat = ops._add_col(m, 1, 2)

        assert np.array_equal(mat, [[0,1,2],[3,1,5],[6,1,8]])
        assert m is mat

        mat = ops._add_col(mat, 0, 2)

        assert np.array_equal(mat, [[0,1,2],[0,1,5],[0,1,8]])
        assert m is mat

    def test_col_flip(self):

        m = np.reshape(range(9), (3,3))
        ops = MatrixOperations()
        mat = ops._flip_cols(m, 1, 2)

        assert np.array_equal(mat, [[0,2,1],[3,5,4],[6,8,7]])
        assert m is mat

