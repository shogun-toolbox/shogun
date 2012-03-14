/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/converter/SupervisedKernelLocalTangentSpaceAlignment.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/KernelLocallyLinearEmbedding.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parallel.h>

using namespace shogun;

class SKLTSA_COVERTREE_POINT
{
public:

	SKLTSA_COVERTREE_POINT(int32_t index, const SGMatrix<float64_t>& dmatrix)
	{
		this->point_index = index;
		this->kernel_matrix = dmatrix;
		this->kii = dmatrix[index*dmatrix.num_rows+index];
	}

	inline double distance(const SKLTSA_COVERTREE_POINT& p) const
	{
		int32_t N = kernel_matrix.num_rows;
		return kii+kernel_matrix[p.point_index*N+p.point_index]-2.0*kernel_matrix[point_index*N+p.point_index];
	}

	inline bool operator==(const SKLTSA_COVERTREE_POINT& p) const
	{
		return (p.point_index==this->point_index);
	}

	int32_t point_index;
	float64_t kii;
	SGMatrix<float64_t> kernel_matrix;
};

CSupervisedKernelLocalTangentSpaceAlignment::CSupervisedKernelLocalTangentSpaceAlignment() :
		CKernelLocalTangentSpaceAlignment(), m_labels(NULL)
{
}

CSupervisedKernelLocalTangentSpaceAlignment::CSupervisedKernelLocalTangentSpaceAlignment(CKernel* kernel, CLabels* labels) :
		CKernelLocalTangentSpaceAlignment(kernel), m_labels(NULL)
{
	set_labels(labels);
}

CSupervisedKernelLocalTangentSpaceAlignment::~CSupervisedKernelLocalTangentSpaceAlignment()
{
	SG_UNREF(m_labels);
}

const char* CSupervisedKernelLocalTangentSpaceAlignment::get_name() const
{ 
	return "SupervisedKernelLocalTangentSpaceAlignment"; 
};

SGMatrix<int32_t> CSupervisedKernelLocalTangentSpaceAlignment::get_neighborhood_matrix(SGMatrix<float64_t> kernel_matrix, int32_t k)
{
	int32_t i;
	int32_t N = kernel_matrix.num_cols;
	ASSERT(m_labels);
	ASSERT(m_labels->get_num_labels()==N);
	
	int32_t* neighborhood_matrix = SG_MALLOC(int32_t, N*k);
	
	float64_t max_dist=0.0;
	for (i=0; i<N; i++)
		max_dist = CMath::max(max_dist,kernel_matrix[i*N+i]);

	std::vector<SKLTSA_COVERTREE_POINT> vectors;
	vectors.reserve(N);
	for (i=0; i<N; i++)
		vectors.push_back(SKLTSA_COVERTREE_POINT(i,kernel_matrix));

	CoverTree<SKLTSA_COVERTREE_POINT>* coverTree = new CoverTree<SKLTSA_COVERTREE_POINT>(2.0*max_dist,vectors);

	for (i=0; i<N; i++)
	{
		std::vector<SKLTSA_COVERTREE_POINT> neighbors = 
		   coverTree->kNearestNeighbors(vectors[i],k+1);

		ASSERT(neighbors.size()>=unsigned(k+1));

		int32_t c = 0;
		for (std::size_t m=1; m<unsigned(k+1); m++)
		{
			if (m_labels->get_int_label(i)==m_labels->get_int_label(neighbors[m].point_index))
			{
				neighborhood_matrix[i*k+c] = neighbors[m].point_index;
				c++;
			}
		}
		SG_PRINT("c=%d",c);
		for (std::size_t m=1; c<k && m<unsigned(k+1); m++)
		{
			if (m_labels->get_int_label(i)!=m_labels->get_int_label(neighbors[m].point_index))
			{
				neighborhood_matrix[i*k+c] = neighbors[m].point_index;
				c++;
			}
		}
	}

	delete coverTree;

	return SGMatrix<int32_t>(neighborhood_matrix,k,N);
}

#endif /* HAVE_LAPACK */
