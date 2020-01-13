from scipy.sparse import diags # can be used with broadcasting of scalars if desired dimensions are large
import numpy as np

# INPUT : diag_broadcast - list of diagonals value to broadcast,length equal to 3 or 5; n - integer, band matrix shape.
# OUTPUT : L - 2D np.ndarray, L.shape[0] depends on bandwidth, L.shape[1] = n-1, do not store main diagonal, where all ones;                  add zeros to the right side of rows to handle with changing length of diagonals.
#          U - 2D np.ndarray, U.shape[0] = n, U.shape[1] depends on bandwidth;
#              add zeros to the bottom of columns to handle with changing length of diagonals.
def band_lu(diag_broadcast, n):
    # enter your code here
    # raise NotImplementedError()
    L_main_diag = np.ones(n)
    L_sub_diag = np.zeros(n-1)
    U_main_diag = np.zeros(n)
    U_upper_diag = np.full(n-1, diag_broadcast[2])
    U_main_diag [0] = diag_broadcast[1]
    try:
        for i in range(1, n):
            if U_main_diag[i-1] == 0:
                raise ZeroDivisionError
            L_sub_diag[i-1] = diag_broadcast[0]/U_main_diag[i-1]
            U_main_diag[i] = diag_broadcast[1]-L_sub_diag[i-1]*diag_broadcast[2]
    except ZeroDivisionError:
        print('zero pivot encountered, LU decomposition does not exist')
        return None, None
    pad_l = np.zeros_like(L_main_diag)
    pad_l[-len(L_sub_diag):] = L_sub_diag
    pad_u = np.zeros_like(U_main_diag)
    pad_u[:len(U_upper_diag)] = U_upper_diag
    return np.vstack([L_main_diag,pad_l]), np.vstack([U_main_diag, pad_u])

arr = [-1, 2 ,-1]
L,U = band_lu(arr, 4)
print('L')
print(L)
print('\nU')
print(U)
print()
arr = [4, -2, 1]
L,U = band_lu(arr, 4)
print('L')
print(L)
print('\nU')
print(U)
