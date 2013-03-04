/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 *
 */

#ifndef TAPKEE_RANDOM_PROJECTION_H_
#define TAPKEE_RANDOM_PROJECTION_H_

namespace tapkee
{
namespace tapkee_internal
{

DenseMatrix gaussian_projection_matrix(IndexType target_dimension, unsigned int current_dimension)
{
	DenseMatrix projection_matrix(target_dimension,current_dimension);

	for (DenseMatrix::Index i=0; i<target_dimension; ++i)
	{
		for (DenseMatrix::Index j=0; j<current_dimension; ++j)
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
