/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{
template class CLinearSolver<float64_t>;
template class CLinearSolver<complex64_t>;
template class CLinearSolver<complex64_t, float64_t>;
}
