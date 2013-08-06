#include <shogun/lib/SGMatrix.h>

#ifdef HAVE_EIGEN3

/**
 * @brief function amari_index 
 * 
 * calculates the amari index for an estimated 
 * and given mixing matrix.
 *
 * @param W the estimated mixing matrix
 * @param A the given mixing matrix
 * @param flag to standardize or not
 *
 */
double amari_index(shogun::SGMatrix<float64_t> W, shogun::SGMatrix<float64_t> A, bool standardize);

#endif //HAVE_EIGEN3
