/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_RANDOM_PROJECTION_H_
#define TAPKEE_RANDOM_PROJECTION_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/tapkee_defines.hpp>
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
			ScalarType v1 = (ScalarType)(rand()+1.f)/((float)RAND_MAX+2.f);
			ScalarType v2 = (ScalarType)(rand()+1.f)/((float)RAND_MAX+2.f);
			ScalarType len = sqrt(-2.f*log(v1));
			projection_matrix(i,j) = len*cos(2.f*M_PI*v2)/sqrt(target_dimension);
		}
	}

	return projection_matrix;
}

}
}

#endif
