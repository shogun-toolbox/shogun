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
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/ProbingSampler.h>
#include <ColPack/ColPackHeaders.h>
#include <gtest/gtest.h>

using namespace std;
using namespace shogun;
using namespace Eigen;
using namespace ColPack;

TEST(ProbingSampler, get_coloring_vector)
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

	// get the sparsity structure and use coloring to get coloring
	SparsityStructure* sp_str=op->get_sparsity_structure(max_pow);

	GraphColoringInterface* Color=new GraphColoringInterface(SRC_MEM_ADOLC,
		sp_str->m_ptr, sp_str->m_num_rows);
	Color->Coloring("NATURAL", "DISTANCE_ONE");

	vector<int32_t> vi_VertexColors;
	Color->GetVertexColors(vi_VertexColors);

	SGVector<int32_t> coloring(vi_VertexColors.size());
	for (vector<int32_t>::iterator it=vi_VertexColors.begin();
		it!=vi_VertexColors.end(); it++)
	{
		coloring[static_cast<int32_t>(distance(vi_VertexColors.begin(), it))]=*it;
	}

	// get the coloring vector using probing sampler
	CProbingSampler* sampler=new CProbingSampler(op, max_pow);
	sampler->precompute();

	SGVector<int32_t> sg_coloring=sampler->get_coloring_vector();

	Map<VectorXi> eig_coloring(coloring.vector, coloring.vlen);
	Map<VectorXi> eig_sg_coloring(sg_coloring.vector, sg_coloring.vlen);

	// should be same
	EXPECT_NEAR((eig_coloring-eig_sg_coloring).cast<float64_t>().norm(), 0.0, 1E-15);

	SG_UNREF(sampler);
	SG_UNREF(op);
	delete Color;
}

TEST(ProbingSampler, probing_samples_big_diag_matrix)
{
	float64_t difficulty=3;
	float64_t min_eigenvalue=0.0001;

	// create a sparse matrix	
	const index_t size=10000;
	SGSparseMatrix<float64_t> sm(size, size);
	CSparseMatrixOperator<float64_t>* op=new CSparseMatrixOperator<float64_t>(sm);
	SG_REF(op);

	// set its diagonal
	SGVector<float64_t> diag(size);
	for (index_t i=0; i<size; ++i)
	{
		diag[i]=CMath::pow(CMath::abs(sg_rand->std_normal_distrib()), difficulty)
			+min_eigenvalue;
	}
	op->set_diagonal(diag);

	CProbingSampler* trace_sampler=new CProbingSampler(op);
	SG_REF(trace_sampler);
	trace_sampler->precompute();

	// test coloring stuffs
	SGVector<int32_t> coloring_vector=trace_sampler->get_coloring_vector();
	EXPECT_EQ(trace_sampler->get_num_samples(), 1);
	for (index_t i=0; i<coloring_vector.vlen; ++i)
		EXPECT_EQ(coloring_vector[i], 0);

	// test that two probing vectors are not equal
	SGVector<float64_t> sample1=trace_sampler->sample(0);
	SGVector<float64_t> sample2=trace_sampler->sample(0);
	Map<VectorXd> sample_1(sample1.vector, sample1.vlen);
	Map<VectorXd> sample_2(sample2.vector, sample2.vlen);
	EXPECT_GT((sample_1-sample_2).norm(), 0.0);


	SG_UNREF(trace_sampler);
	SG_UNREF(op);
}

TEST(ProbingSampler, mean_variance)
{
	const index_t size=1000;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	for (index_t i=0; i<size; ++i)
		m(i,i)=1;
	for (index_t i=0; i<size-1; ++i)
		m(i,i+1)=1;
	for (index_t i=0; i<size-1; ++i)
		m(i+1,i)=1;

	CSparseFeatures<float64_t>* feat=new CSparseFeatures<float64_t>(m);
	SGSparseMatrix<float64_t> sm=feat->get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t>* A=new CSparseMatrixOperator<float64_t>(sm);

	CProbingSampler* trace_sampler=new CProbingSampler(A);
	trace_sampler->precompute();

	index_t num_samples=trace_sampler->get_num_samples();	
	for (index_t i=0; i<num_samples; ++i)
	{
		const SGVector<float64_t>& sample=trace_sampler->sample(i);
		EXPECT_NEAR(CStatistics::mean(sample), 0.0, 0.1);
		EXPECT_NEAR(CStatistics::variance(sample), 1.0/num_samples, 0.01);
	}
}
#endif // HAVE_EIGEN3
#endif // HAVE_COLPACK
