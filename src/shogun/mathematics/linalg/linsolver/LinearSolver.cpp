/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Björn Esser
 */

#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{
template class CLinearSolver<float64_t>;
template class CLinearSolver<complex128_t>;
template class CLinearSolver<complex128_t, float64_t>;
}
