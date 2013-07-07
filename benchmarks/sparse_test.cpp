/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/Time.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

SGVector<float64_t> sg_m_apply(SGSparseMatrix<float64_t> m, SGVector<float64_t> v)
{
	SGVector<float64_t> r(v.vlen);
	ASSERT(v.vlen==m.num_vectors);
#pragma omp parallel for
	for (index_t i=0; i<m.num_vectors; ++i)
		r[i]=m[i].dense_dot(1.0, v.vector, v.vlen, 0.0);

	return r;
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
	//sg_io->set_loglevel(MSG_GCDEBUG);

	const index_t n=100;
	const index_t times=5;
	const index_t size=1000000;
	SGVector<float64_t> v(size);
	v.set_const(1.0);
	Map<VectorXd> map_v(v.vector, v.vlen);
	CTime time;

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
			// the first col
			rest[i]=SG_MALLOC(Entry, 2);
			rest[i][0].feat_index=0; 
			rest[i][0].entry=0.01;

			// the diagonal element
			rest[i][1].feat_index=i+1; 
			rest[i][1].entry=1.836593;

			vec[i+1].features=rest[i];
			vec[i+1].num_feat_entries=2;

			sg_m[i+1]=vec[i+1].get();
		}
		SGVector<float64_t> r(size);

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
