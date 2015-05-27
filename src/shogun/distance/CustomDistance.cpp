/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/base/Parameter.h>
#include <shogun/lib/common.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/DummyFeatures.h>
#include <shogun/io/SGIO.h>

using namespace distance;

CCustomDistance::CCustomDistance() : CDistance()
{
	init();
}

CCustomDistance::CCustomDistance(CDistance* d) : CDistance()
{
	init();

	if (d->lhs_equals_rhs())
	{
		int32_t cols=d->get_num_vec_lhs();
		SG_DEBUG("using custom distance of size %dx%d\n", cols,cols)

		dmatrix= SG_MALLOC(float32_t, int64_t(cols)*(cols+1)/2);

		upper_diagonal=true;
		num_rows=cols;
		num_cols=cols;

		for (int32_t row=0; row<num_rows; row++)
		{
			for (int32_t col=row; col<num_cols; col++)
				dmatrix[int64_t(row) * num_cols - int64_t(row)*(row+1)/2 + col]=d->distance(row,col);
		}
	}
	else
	{
		int32_t rows=d->get_num_vec_lhs();
		int32_t cols=d->get_num_vec_rhs();
		dmatrix= SG_MALLOC(float32_t, int64_t(rows)*cols);

		upper_diagonal=false;
		num_rows=rows;
		num_cols=cols;

		for (int32_t row=0; row<num_rows; row++)
		{
			for (int32_t col=0; col<num_cols; col++)
				dmatrix[int64_t(row) * num_cols + col]=d->distance(row,col);
		}
	}

	dummy_init(num_rows, num_cols);
}

CCustomDistance::CCustomDistance(const SGMatrix<float64_t> distance_matrix)
: CDistance()
{
	init();
	set_full_distance_matrix_from_full(distance_matrix.matrix,
	                                   distance_matrix.num_rows,
	                                   distance_matrix.num_cols);
}

CCustomDistance::CCustomDistance(const float64_t* dm, int32_t rows, int32_t cols)
: CDistance()
{
	init();
	set_full_distance_matrix_from_full(dm, rows, cols);
}

CCustomDistance::CCustomDistance(const float32_t* dm, int32_t rows, int32_t cols)
: CDistance()
{
	init();
	set_full_distance_matrix_from_full(dm, rows, cols);
}

CCustomDistance::~CCustomDistance()
{
	cleanup();
}

bool CCustomDistance::dummy_init(int32_t rows, int32_t cols)
{
	return init(new CDummyFeatures(rows), new CDummyFeatures(cols));
}

bool CCustomDistance::init(CFeatures* l, CFeatures* r)
{
	CDistance::init(l, r);

	SG_DEBUG("num_vec_lhs: %d vs num_rows %d\n", l->get_num_vectors(), num_rows)
	SG_DEBUG("num_vec_rhs: %d vs num_cols %d\n", r->get_num_vectors(), num_cols)
	ASSERT(l->get_num_vectors()==num_rows)
	ASSERT(r->get_num_vectors()==num_cols)
	return true;
}


void CCustomDistance::cleanup_custom()
{
	SG_DEBUG("cleanup up custom distance\n")
	SG_FREE(dmatrix);
	dmatrix=NULL;
	upper_diagonal=false;
	num_cols=0;
	num_rows=0;
}

void CCustomDistance::init()
{
	dmatrix=NULL;
	num_rows=0;
	num_cols=0;
	upper_diagonal=false;

	m_parameters->add_matrix(&dmatrix, &num_rows, &num_cols, "dmatrix", "Distance Matrix");
	m_parameters->add(&upper_diagonal, "upper_diagonal", "Upper diagonal");
}

void CCustomDistance::cleanup()
{
	cleanup_custom();
}

float64_t CCustomDistance::compute(int32_t row, int32_t col)
{
	ASSERT(dmatrix)

	if (upper_diagonal)
	{
		if (row <= col)
		{
			int64_t r=row;
			return dmatrix[r*num_cols - r*(r+1)/2 + col];
		}
		else
		{
			int64_t c=col;
			return dmatrix[c*num_cols - c*(c+1)/2 + row];
		}
	}
	else
	{
		int64_t r=row;
		return dmatrix[r*num_cols+col];
	}
}
