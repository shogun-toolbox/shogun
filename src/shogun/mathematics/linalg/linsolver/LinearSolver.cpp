/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Bjoern Esser
 */

#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{
template class LinearSolver<float64_t>;
template class LinearSolver<complex128_t>;
template class LinearSolver<complex128_t, float64_t>;
}
