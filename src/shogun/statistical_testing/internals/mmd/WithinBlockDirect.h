/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2014  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http:/www.gnu.org/licenses/>.
 */

#ifndef WITHIN_BLOCK_DIRECT_H_
#define WITHIN_BLOCK_DIRECT_H_

#include <shogun/lib/common.h>

namespace shogun
{

template <typename T> class SGMatrix;
template <typename T> class CGPUMatrix;

namespace internal
{

namespace mmd
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct WithinBlockDirect
{
	typedef float32_t return_type;
	return_type operator()(const SGMatrix<return_type>& kernel_matrix);
//	return_type operator()(const CGPUMatrix<return_type>& kernel_matrix);
};
#endif // DOXYGEN_SHOULD_SKIP_THIS
}

}

}

#endif // WITHIN_BLOCK_DIRECT_H_
