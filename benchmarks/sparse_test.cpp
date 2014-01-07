/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <lib/common.h>

#ifdef HAVE_EIGEN3
#include <lib/Time.h>
#include <lib/SGVector.h>
#include <lib/SGSparseMatrix.h>
#include <lib/SGSparseVector.h>
#include <mathematics/Math.h>
#include <mathematics/eigen3.h>
#include <pthread.h>

using namespace shogun;
using namespace Eigen;

struct APPLY_THREAD_PARAM
{
	int32_t start;
	int32_t stop;
	float64_t* result;
	float64_t* vec;
	int32_t len;
	SGSparseVector<float64_t>* sm;
};


int32_t get_nnz(SGSparseMatrix<float64_t> m)
{

	int32_t nnz=0;
	int32_t n=m.num_vectors;

	for (int i=0; i<n; i++)
	{
		nnz+=m[i].num_feat_entries;
	}
	return nnz;
}

static void* dot_helper(void* p)
{
	APPLY_THREAD_PARAM* par=(APPLY_THREAD_PARAM*) p;
	float64_t* r = par->result;
	SGSparseVector<float64_t>* m=par->sm;
	float64_t* vec = par->vec;
	int32_t len = par->len;
	int32_t start = par->start;
	int32_t stop = par->stop;

	for (index_t i=start; i<stop; ++i)
		r[i]=m[i].dense_dot(1.0, vec, len, 0.0);
}


SGVector<float64_t> sg_m_apply(SGSparseMatrix<float64_t> m, SGVector<float64_t> v)
{
	SGVector<float64_t> r(v.vlen);
	ASSERT(v.vlen==m.num_vectors);

	int num_threads=8;
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
	APPLY_THREAD_PARAM* params = SG_MALLOC(APPLY_THREAD_PARAM, num_threads);
	int32_t step= m.num_vectors/num_threads;

	int32_t start=0;
	int32_t stop=m.num_vectors;
	int32_t t;

	for (t=0; t<num_threads-1; t++)
	{
		params[t].start = start+t*step;
		params[t].stop = start+(t+1)*step;
		params[t].result = r.vector;
		params[t].sm=m.sparse_matrix;
		params[t].vec=v.vector;
		params[t].len=v.vlen;
		pthread_create(&threads[t], NULL,
				dot_helper, (void*)&params[t]);
	}

	params[t].start = start+t*step;
	params[t].stop = stop;
	params[t].result = r.vector;
	params[t].sm=m.sparse_matrix;
	params[t].vec=v.vector;
	params[t].len=v.vlen;
	dot_helper((void*) &params[t]);

	for (t=0; t<num_threads-1; t++)
		pthread_join(threads[t], NULL);

	SG_FREE(params);
	SG_FREE(threads);

	return r;
}

int main(int argc, char** argv)
{
	Eigen::initParallel();
	init_shogun_with_defaults();
	//sg_io->set_loglevel(MSG_GCDEBUG);

	const index_t n=100;
	const index_t times=5;
	const index_t size=1000000;
	SGVector<float64_t> v(size);
	v.set_const(1.0);
	Map<VectorXd> map_v(v.vector, v.vlen);
	CTime time;
	CMath::init_random(17);

	SG_SPRINT("time\tshogun (s)\teigen3 (s)\n\n");
	for (index_t t=0; t<times; ++t)
	{
//#ifdef RUN_SHOGUN
		SGSparseMatrix<float64_t> sg_m(size, size);
		typedef SGSparseVectorEntry<float64_t> Entry;
		SGSparseVector<float64_t> *vec=SG_MALLOC(SGSparseVector<float64_t>, size);

		// for first row
		Entry *first=SG_MALLOC(Entry, size);
		// the digonal index for row #1
		first[0].feat_index=0;
		first[0].entry=1.836593;
		for (index_t i=1; i<size; ++i)
		{
			// fill the index for row #1
			first[i].feat_index=i;
			first[i].entry=0.02;
		}
		vec[0].features=first;
		vec[0].num_feat_entries=size;
		sg_m[0]=vec[0].get();

		// fill the rest of the rows
		Entry** rest=SG_MALLOC(Entry*, size-1);
		for (index_t i=0; i<size-1; ++i)
		{
			int num=40;
			// the first col
			rest[i]=SG_MALLOC(Entry, num);

			for (int j=0; j<i && j<num; j++)
			{
				rest[i][j].feat_index=j;
				rest[i][j].entry=0.01+j;
			}

			if (i>num)
			{
				//// the diagonal element
				rest[i][num-1].feat_index=i+1;
				rest[i][num-1].entry=1.836593;
			}

			vec[i+1].features=rest[i];
			vec[i+1].num_feat_entries=num;

			sg_m[i+1]=vec[i+1].get();
		}
		SGVector<float64_t> r(size);

		SG_SPRINT("nnz=%d\n", get_nnz(sg_m));

		// sg starts
		time.start();
		for (index_t i=0; i<n; ++i)
			r=sg_m_apply(sg_m, v);
		float64_t sg_time = time.cur_time_diff();

		Map<VectorXd> map_r(r.vector, r.vlen);
		float64_t sg_norm=map_r.norm();

//#endif // RUN_SHOGUN

//#ifdef RUN_EIGEN
		const SparseMatrix<float64_t> &eig_m=EigenSparseUtil<float64_t>::toEigenSparse(sg_m);
		VectorXd eig_r(size);

		// eigen3 starts
		time.start();
		for (index_t i=0; i<n; ++i)
			eig_r=eig_m*map_v;

		float64_t eig_time = time.cur_time_diff();
		float64_t eig_norm=eig_r.norm();
//#endif // RUN_EIGEN

		SG_SPRINT("%d\t%lf\t%lf\n", t, sg_time, eig_time);
		//ASSERT(sg_time>eig_time);
		ASSERT(CMath::abs(sg_norm-eig_norm)<=CMath::MACHINE_EPSILON)

		SG_FREE(vec);
		SG_FREE(rest);
	}


	exit_shogun();

	return 0;
}
#endif // HAVE_EIGEN3
