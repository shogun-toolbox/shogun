/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_RandomProjection_H_
#define TAPKEE_RandomProjection_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
#include <shogun/lib/tapkee/utils/time.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

DenseMatrix gaussian_projection_matrix(IndexType target_dimension, IndexType current_dimension)
{
	DenseMatrix projection_matrix(target_dimension,current_dimension);

	for (IndexType i=0; i<target_dimension; ++i)
	{
		for (IndexType j=0; j<current_dimension; ++j)
		{
			projection_matrix(i,j) = tapkee::gaussian_random()/sqrt(target_dimension);
		}
	}

	return projection_matrix;
}

}
}

#endif
