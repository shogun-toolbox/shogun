/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Sergey Lisitsyn,
 *          Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/DummyFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CustomDistance::CustomDistance() : Distance()
{
	init();
}

CustomDistance::CustomDistance(const std::shared_ptr<Distance>& d) : Distance()
{
	init();

	if (d->lhs_equals_rhs())
	{
		int32_t cols=d->get_num_vec_lhs();
		SG_DEBUG("using custom distance of size {}x{}", cols,cols)

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

CustomDistance::CustomDistance(const SGMatrix<float64_t> distance_matrix)
: Distance()
{
	init();
	set_full_distance_matrix_from_full(distance_matrix.matrix,
	                                   distance_matrix.num_rows,
	                                   distance_matrix.num_cols);
}

CustomDistance::CustomDistance(const float64_t* dm, int32_t rows, int32_t cols)
: Distance()
{
	init();
	set_full_distance_matrix_from_full(dm, rows, cols);
}

CustomDistance::CustomDistance(const float32_t* dm, int32_t rows, int32_t cols)
: Distance()
{
	init();
	set_full_distance_matrix_from_full(dm, rows, cols);
}

CustomDistance::~CustomDistance()
{
	cleanup();
}

bool CustomDistance::dummy_init(int32_t rows, int32_t cols)
{
	return init(std::make_shared<DummyFeatures>(rows), std::make_shared<DummyFeatures>(cols));
}

bool CustomDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Distance::init(l, r);

	SG_DEBUG("num_vec_lhs: {} vs num_rows {}", l->get_num_vectors(), num_rows)
	SG_DEBUG("num_vec_rhs: {} vs num_cols {}", r->get_num_vectors(), num_cols)
	ASSERT(l->get_num_vectors()==num_rows)
	ASSERT(r->get_num_vectors()==num_cols)
	return true;
}


void CustomDistance::cleanup_custom()
{
	SG_DEBUG("cleanup up custom distance")
	SG_FREE(dmatrix);
	dmatrix=NULL;
	upper_diagonal=false;
	num_cols=0;
	num_rows=0;
}

void CustomDistance::init()
{
	dmatrix=NULL;
	num_rows=0;
	num_cols=0;
	upper_diagonal=false;

	/*m_parameters->add_matrix(&dmatrix, &num_rows, &num_cols, "dmatrix", "Distance Matrix")*/;
	watch_param(
	    "dmatrix", &dmatrix, &num_rows, &num_cols,
	    AnyParameterProperties("Distance Matrix"));

	SG_ADD(
	    &upper_diagonal, "upper_diagonal", "Upper diagonal");
}

void CustomDistance::cleanup()
{
	cleanup_custom();
}

float64_t CustomDistance::compute(int32_t row, int32_t col)
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
