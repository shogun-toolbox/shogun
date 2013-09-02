/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>

#ifdef HAVE_COLPACK
#ifdef HAVE_EIGEN3

#include <vector>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>
#include <shogun/mathematics/logdet/ProbingSampler.h>
#include <ColPack/ColPackHeaders.h>
#include <gtest/gtest.h>

using namespace std;
using namespace shogun;
using namespace Eigen;
using namespace ColPack;

TEST(ProbingSampler, get_probing_vector)
{
	const int32_t size=9;
	const int32_t max_pow=10;

	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);
	for (int32_t i=0; i<size; ++i)
		m(i,i)=2.0;

	for (int32_t i=0; i<size; i+=4)
		m(i,size-1)=2.0;

	for (int32_t i=0; i<size; i+=4)
		m(size-1,i)=2.0;

	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> sm=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t>* op
		=new CSparseMatrixOperator<float64_t>(sm);
	SG_REF(op);

	// get the sparsity structure and use coloring to get probing
	SparsityStructure* sp_str=op->get_sparsity_structure(max_pow);

	GraphColoringInterface* Color=new GraphColoringInterface(SRC_MEM_ADOLC,
		sp_str->m_ptr, sp_str->m_num_rows);
	Color->Coloring("NATURAL", "DISTANCE_ONE");

	vector<int32_t> vi_VertexColors;
	Color->GetVertexColors(vi_VertexColors);

	SGVector<int32_t> probing(vi_VertexColors.size());
	for (vector<int32_t>::iterator it=vi_VertexColors.begin();
		it!=vi_VertexColors.end(); it++)
	{
		probing[static_cast<int32_t>(distance(vi_VertexColors.begin(), it))]=*it;
	}

	// get the probing vector using probing sampler
	CProbingSampler* sampler=new CProbingSampler(op, max_pow);
	sampler->precompute();

	SGVector<int32_t> sg_probing=sampler->get_probing_vector();

	Map<VectorXi> eig_probing(probing.vector, probing.vlen);
	Map<VectorXi> eig_sg_probing(sg_probing.vector, sg_probing.vlen);

	// should be same
	EXPECT_NEAR((eig_probing-eig_sg_probing).cast<float64_t>().norm(), 0.0, 1E-15);

	SG_UNREF(sampler);
	SG_UNREF(op);
	delete Color;
}
#endif // HAVE_EIGEN3
#endif // HAVE_COLPACK
