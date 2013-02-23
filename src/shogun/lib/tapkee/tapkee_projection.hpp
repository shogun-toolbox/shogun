/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando J. Iglesias García
 *
 */

namespace tapkee
{

struct ProjectionImplementation
{
	virtual ~ProjectionImplementation()
	{
	}
	virtual DenseVector project(const DenseVector& vec) = 0;
};

struct ProjectingFunction
{
	ProjectingFunction() : implementation(NULL) {};
	ProjectingFunction(ProjectionImplementation* impl) : implementation(impl) {};
	void clear() 
	{ 
		delete implementation;
	}
	inline DenseVector operator()(const DenseVector& vec)
	{
		return implementation->project(vec);
	}
	ProjectionImplementation* implementation;
};

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

