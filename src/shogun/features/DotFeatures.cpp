/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni, Michele Mazzoni, Viktor Gal,
 *          Fernando Iglesias, Evgeniy Andreev, Evan Shelhamer, Alesis Novik,
 *          Sergey Lisitsyn
 */

#include <shogun/base/Parallel.h>
#include <shogun/base/progress.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

using namespace shogun;


DotFeatures::DotFeatures(int32_t size)
	:Features(size)
{
	init();
}


DotFeatures::DotFeatures(const DotFeatures & orig)
	:Features(orig)
{
	init();
}


DotFeatures::DotFeatures(std::shared_ptr<File> loader)
	:Features(loader)
{
	init();
}

void DotFeatures::dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b) const
{
	ASSERT(output)
	ASSERT(start>=0)
	ASSERT(start<stop)
	ASSERT(stop<=get_num_vectors())

	int32_t num_vectors=stop-start;
	ASSERT(num_vectors>0)
	SGVector<float64_t> sgvec(vec, dim, false);

	int32_t num_threads;
	int32_t step;
	auto pb = SG_PROGRESS(range(num_vectors));
#pragma omp parallel shared(num_threads, step)
	{
#ifdef HAVE_OPENMP
		#pragma omp single
		{
			num_threads=omp_get_num_threads();
			step=num_vectors/num_threads;
			num_threads--;
		}
		int32_t thread_num=omp_get_thread_num();
#else
		num_threads=0;
		step=num_vectors;
		int32_t thread_num=0;
#endif

		int32_t t_start=thread_num*step;
		int32_t t_stop=(thread_num==num_threads) ? num_vectors : (thread_num+1)*step;

#ifdef WIN32
		for (int32_t i=t_start; i<t_stop; i++)
#else
		// TODO: replace with the new signal
		// for (int32_t i=t_start; i<t_stop &&
		//		!Signal::cancel_computations(); i++)
		for (int32_t i = t_start; i < t_stop; i++)
#endif
		{
			if (alphas)
				output[i]=alphas[i]*this->dot(i + start, sgvec)+b;
			else
				output[i]=this->dot(i + start, sgvec)+b;
			pb.print_progress();
		}
	}
	pb.complete();
}

void DotFeatures::dense_dot_range_subset(int32_t* sub_index, int32_t num, float64_t* output, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b) const
{
	ASSERT(sub_index)
	ASSERT(output)

	SGVector<float64_t> sgvec(vec, dim, false);
	auto pb = SG_PROGRESS(range(num));
	int32_t num_threads;
	int32_t step;
	#pragma omp parallel shared(num_threads, step)
	{
#ifdef HAVE_OPENMP
		#pragma omp single
		{
			num_threads=omp_get_num_threads();
			step=num/num_threads;
			num_threads--;
		}
		int32_t thread_num=omp_get_thread_num();
#else
		num_threads=0;
		step = num;
		int32_t thread_num=0;
#endif

		int32_t t_start=thread_num*step;
		int32_t t_stop=(thread_num==num_threads) ? num : (thread_num+1)*step;

#ifdef WIN32
		for (int32_t i=t_start; i<t_stop; i++)
#else
		// TODO: replace with the new signal
		// for (int32_t i=t_start; i<t_stop &&
		//		!Signal::cancel_computations(); i++)
		for (int32_t i = t_start; i < t_stop; i++)
#endif
		{
			if (alphas)
				output[i]=alphas[sub_index[i]]*this->dot(sub_index[i], sgvec)+b;
			else
				output[i]=this->dot(sub_index[i], sgvec)+b;
			pb.print_progress();
		}
	}
	pb.complete();
}

SGMatrix<float64_t> DotFeatures::get_computed_dot_feature_matrix() const
{

	int64_t offs=0;
	int32_t num=get_num_vectors();
	int32_t dim=get_dim_feature_space();
	ASSERT(num>0)
	ASSERT(dim>0)

	SGMatrix<float64_t> m(dim, num);
	m.zero();

	for (int32_t i=0; i<num; i++)
	{
		add_to_dense_vec(1.0, i, &(m.matrix[offs]), dim);
		offs+=dim;
	}

	return m;
}

SGVector<float64_t> DotFeatures::get_computed_dot_feature_vector(int32_t num) const
{

	int32_t dim=get_dim_feature_space();
	ASSERT(num>=0 && num<=get_num_vectors())
	ASSERT(dim>0)

	SGVector<float64_t> v(dim);
	v.zero();
	add_to_dense_vec(1.0, num, v.vector, dim);
	return v;
}

SGVector<float64_t> DotFeatures::get_mean() const
{
	int32_t num=get_num_vectors();
	int32_t dim=get_dim_feature_space();
	ASSERT(num>0)
	ASSERT(dim>0)

	SGVector<float64_t> mean(dim);
	linalg::zero(mean);

	for (int32_t i = 0; i < num; ++i)
		add_to_dense_vec(1, i, mean.vector, dim);

	linalg::scale(mean, mean, 1.0 / num);

	return mean;
}

SGVector<float64_t> DotFeatures::get_std(bool colwise) const
{
	auto num=get_num_vectors();
	auto dim=get_dim_feature_space();
	ASSERT(num>0)
	ASSERT(dim>0)

	auto mean = get_mean();
	if (!colwise)
	{
		auto global_mean = linalg::sum(mean) / dim;
		linalg::set_const(mean, global_mean);
	}

	linalg::scale(mean, mean, -1.0);
	SGVector<float64_t> std(dim);
	for (index_t i = 0; i < num; ++i)
	{
		auto m = mean.clone();
		add_to_dense_vec(1, i, m.vector, dim);
		linalg::element_prod(m, m, m);
		linalg::add(m, std, std);
	}

	if (!colwise)
	{
		SGVector<float64_t> global_std(1);
		global_std[0] = std::sqrt(linalg::sum(std) / (num*dim));
		return global_std;
	}

	linalg::scale(std, std, 1.0 / num);
	for (index_t i = 0; i < num; ++i)
		std[i] = std::sqrt(std[i]);

	return std;
}

SGVector<float64_t>
DotFeatures::compute_mean(std::shared_ptr<DotFeatures> lhs, std::shared_ptr<DotFeatures> rhs)
{
	ASSERT(lhs && rhs)
	ASSERT(lhs->get_dim_feature_space() == rhs->get_dim_feature_space())

	int32_t num_lhs=lhs->get_num_vectors();
	int32_t num_rhs=rhs->get_num_vectors();
	int32_t dim=lhs->get_dim_feature_space();
	ASSERT(num_lhs>0)
	ASSERT(num_rhs>0)
	ASSERT(dim>0)

	SGVector<float64_t> mean(dim);
	linalg::zero(mean);

	for (int i = 0; i < num_lhs; i++)
		lhs->add_to_dense_vec(1, i, mean.vector, dim);

	for (int i = 0; i < num_rhs; i++)
		rhs->add_to_dense_vec(1, i, mean.vector, dim);

	linalg::scale(mean, mean, 1.0 / (num_lhs + num_rhs));

	return mean;
}

SGMatrix<float64_t> DotFeatures::get_cov(bool copy_data_for_speed) const
{
	int32_t num=get_num_vectors();
	int32_t dim=get_dim_feature_space();
	ASSERT(num>0)
	ASSERT(dim>0)

	SGMatrix<float64_t> cov(dim, dim);
	SGVector<float64_t> mean = get_mean();

	if (copy_data_for_speed)
	{
		SGMatrix<float64_t> centered_data(dim, num);
		for (int i = 0; i < num; i++)
		{
			SGVector<float64_t> v = get_computed_dot_feature_vector(i);
			centered_data.set_column(i, linalg::add(v, mean, 1.0, -1.0));
		}

		cov = linalg::matrix_prod(centered_data, centered_data, false, true);
	}
	else
	{
		linalg::zero(cov);
		for (int i = 0; i < num; i++)
		{
			SGVector<float64_t> v = get_computed_dot_feature_vector(i);
			linalg::add(v, mean, v, 1.0, -1.0);
			for (int m = 0; m < v.vlen; m++)
				linalg::add_col_vec(cov, m, v, cov, 1.0, v.vector[m]);
		}
		for (int m = 0; m < dim - 1; m++)
		{
			for (int n = m + 1; n < dim; n++)
			{
				(cov.matrix)[m * dim + n] = (cov.matrix)[n * dim + m];
			}
		}
	}
	linalg::scale(cov, cov, 1.0 / num);

	return cov;
}

SGMatrix<float64_t> DotFeatures::compute_cov(
    std::shared_ptr<DotFeatures> lhs, std::shared_ptr<DotFeatures> rhs, bool copy_data_for_speed)
{
	std::shared_ptr<DotFeatures> feats[2];
	feats[0]=lhs;
	feats[1]=rhs;

	int32_t nums[2], dims[2], num=0;

	for (int i = 0; i < 2; i++)
	{
		nums[i]=feats[i]->get_num_vectors();
		dims[i]=feats[i]->get_dim_feature_space();
		ASSERT(nums[i]>0)
		ASSERT(dims[i]>0)
		num += nums[i];
	}

	ASSERT(dims[0]==dims[1])
	int32_t dim = dims[0];

	SGMatrix<float64_t> cov(dim, dim);
	SGVector<float64_t> mean = compute_mean(lhs, rhs);

	if (copy_data_for_speed)
	{
		SGMatrix<float64_t> centered_data(dim, num);
		for (int i = 0; i < num; i++)
		{
			SGVector<float64_t> v =
			    i < nums[0] ? lhs->get_computed_dot_feature_vector(i)
			                : rhs->get_computed_dot_feature_vector(i - nums[0]);

			centered_data.set_column(i, linalg::add(v, mean, 1.0, -1.0));
		}

		cov = linalg::matrix_prod(centered_data, centered_data, false, true);
	}
	else
	{
		linalg::zero(cov);
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < nums[i]; j++)
			{
				SGVector<float64_t> v =
				    feats[i]->get_computed_dot_feature_vector(j);
				linalg::add(v, mean, v, 1.0, -1.0);
				for (int m = 0; m < v.vlen; m++)
					linalg::add_col_vec(cov, m, v, cov, 1.0, v.vector[m]);
			}
		}

		for (int m = 0; m < dim - 1; m++)
		{
			for (int n = m + 1; n < dim; n++)
			{
				(cov.matrix[m * dim + n]) = (cov.matrix)[n * dim + m];
			}
		}
	}
	linalg::scale(cov, cov, 1.0 / num);

	return cov;
}

void DotFeatures::init()
{
	set_property(FP_DOT);
}
