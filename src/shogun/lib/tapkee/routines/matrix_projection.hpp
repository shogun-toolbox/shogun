/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_MATRIX_PROJECTION_H_
#define TAPKEE_MATRIX_PROJECTION_H_

namespace tapkee
{
namespace tapkee_internal
{

struct MatrixProjectionImplementation : public ProjectionImplementation
{
	MatrixProjectionImplementation(DenseMatrix matrix) : mat(matrix) 
	{
	}

	virtual ~MatrixProjectionImplementation()
	{
	}

	virtual DenseVector project(const DenseVector& vec) 
	{
		return mat*vec;
	}

	DenseMatrix mat;
};

}
}
#endif
