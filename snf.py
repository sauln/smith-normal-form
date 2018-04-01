import numpy as np


class MatrixOperations():
    def __init__(self, mod=2):
        self.__mod = mod

    def _add_row(self, mat, i, j, r=1):
        mat[i, :] = (mat[i, :] + r * mat[j, :]) % self.__mod
        return mat

    def _flip_rows(self, mat, i, j):
        mat[[i,j]] = mat[[j,i]] 
        return mat

    def _add_col(self, mat, i, j, r=1):
        mat[:, i] = (mat[:, i] + r * mat[:, j] ) % self.__mod
        return mat  

    def _flip_cols(self, mat, i, j):
        mat[:, [i,j]] = mat[:, [j,i]] 
        return mat

class SmithNormalForm(MatrixOperations):
    def __init__(self, modulo=2, inplace=False):
        super().__init__(modulo)

        self._inplace = inplace

    def smithify(self, mat, i=0):
        """
            This algorithm works in Z_2 only:
            
            if \E j\ge i, k \ge i with $B_jk = 1:
                R_j <=> R_i; C_k <=> C_i

            for h = i+1 to m
                if B_hi = 1:
                    R_h + R_i

            for l = i+1 to n:
                if B_il = 1:
                    C_l + C_i
        """
    
        # Have this wrapper so we can copy the matrix the first time around

        if self._inplace:
            bm = mat
        else:
            bm = np.copy(mat)
            bm = self.do_smithing(bm, i)

        return bm

    def do_smithing(self, mat, i):
        m, n = mat.shape
        
        found_ones = np.where(mat[i:, i:] == 1)

        if found_ones[0].size:
            j,k = found_ones[0][0]+i, found_ones[1][0]+i

            if (j,k) != (i,i):
                mat = self._flip_rows(mat, i, j)
                mat = self._flip_cols(mat, i, k)

            for h in range(i+1, m):
                # Zero out row
                if mat[h,i] == 1:
                    mat = self._add_row(mat, h, i)
        
            for l in range(i+1, n):
                # Zero out column
                if mat[i,l] == 1:
                    mat = self._add_col(mat, l, i)
            
            mat = self.do_smithing(mat, i+1)
    
        return mat