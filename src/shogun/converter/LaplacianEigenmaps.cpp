/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/LaplacianEigenmaps.h>
#include <shogun/converter/EmbeddingConverter.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CLaplacianEigenmaps::CLaplacianEigenmaps() :
		CEmbeddingConverter()
{
	m_k = 3;
	m_tau = 1.0;
	
	init();
}

void CLaplacianEigenmaps::init()
{
	m_parameters->add(&m_k, "k", "number of neighbors");
	m_parameters->add(&m_tau, "tau", "heat distribution coefficient");
}

CLaplacianEigenmaps::~CLaplacianEigenmaps()
{
}

void CLaplacianEigenmaps::set_k(int32_t k)
{
	ASSERT(k>0);
	m_k = k;
}

int32_t CLaplacianEigenmaps::get_k() const
{
	return m_k;
}

void CLaplacianEigenmaps::set_tau(float64_t tau)
{
	m_tau = tau;
}

float64_t CLaplacianEigenmaps::get_tau() const
{
	return m_tau;
}

const char* CLaplacianEigenmaps::get_name() const 
{ 
	return "LaplacianEigenmaps";
};

CFeatures* CLaplacianEigenmaps::apply(CFeatures* features)
{
	// shorthand for simplefeatures
	SG_REF(features);

	// get dimensionality and number of vectors of data
	int32_t N = features->get_num_vectors();
	ASSERT(m_k<N);
	ASSERT(m_target_dim<N);

	// compute distance matrix
	ASSERT(m_distance);
	m_distance->init(features,features);
	CSimpleFeatures<float64_t>* embedding = embed_distance(m_distance);
	m_distance->remove_lhs_and_rhs();
	SG_UNREF(features);
	return (CFeatures*)embedding;
}

CSimpleFeatures<float64_t>* CLaplacianEigenmaps::embed_distance(CDistance* distance)
{
	int32_t i,j;
	SGMatrix<float64_t> W_sgmatrix = distance->get_distance_matrix();
	ASSERT(W_sgmatrix.num_rows==W_sgmatrix.num_cols);
	int32_t N = W_sgmatrix.num_rows;
	// shorthand
	float64_t* W_matrix = W_sgmatrix.matrix;

	// init heap to use
	CFibonacciHeap* heap = new CFibonacciHeap(N);
	float64_t tmp;
	// for each object
	for (i=0; i<N; i++)
	{
		// fill heap
		for (j=0; j<N; j++)
			heap->insert(j,W_matrix[i*N+j]);

		// rearrange heap with extracting ith object itself
		heap->extract_min(tmp);

		// extract nearest neighbors, takes ~O(k*log n), and change sign for them
		for (j=0; j<m_k; j++)
			W_matrix[i*N+heap->extract_min(tmp)] *= -1.0;

		// remove all 'positive' distances and change 'negative' ones to positive
		for (j=0; j<N; j++)
		{
			if (W_matrix[i*N+j]>0.0)
				W_matrix[i*N+j] = 0.0;
			else
				W_matrix[i*N+j] *= -1.0;
		}
		
		// clear heap to reuse
		heap->clear();
	}
	delete heap;
	// make distance matrix symmetric with mutual kNN relation
	for (i=0; i<N; i++)
	{
		// check only upper triangle
		for (j=i; j<N; j++)
		{
			// make kNN relation symmetric
			if (W_matrix[i*N+j]!=0.0 || W_matrix[j*N+i]==0.0)
			{
				W_matrix[j*N+i] = W_matrix[i*N+j];
			}
			if (W_matrix[j*N+i]!=0.0 || W_matrix[i*N+j]==0.0)
			{
				W_matrix[i*N+j] = W_matrix[j*N+i];
			}
			
			if (W_matrix[i*N+j] != 0.0)
			{
				// compute heat, exp(-d^2/tau)
				tmp = CMath::exp(-CMath::sq(W_matrix[i*N+j])/m_tau);
				W_matrix[i*N+j] = tmp;
				W_matrix[j*N+i] = tmp;
			}
		}
	}

	// compute D
	CSimpleFeatures<float64_t>* embedding = construct_embedding(W_sgmatrix);
	W_sgmatrix.destroy_matrix();

	return embedding;
}

CSimpleFeatures<float64_t>* CLaplacianEigenmaps::construct_embedding(SGMatrix<float64_t> W_matrix)
{
	int32_t i,j;
	int32_t N = W_matrix.num_cols;

	float64_t* D_diag_vector = SG_CALLOC(float64_t, N);
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
			D_diag_vector[i] += W_matrix[i*N+j];
	}

	// W = -W
	for (i=0; i<N*N; i++)
		if (W_matrix[i]>0.0)
			W_matrix[i] *= -1.0;
	// W = W + D
	for (i=0; i<N; i++)
		W_matrix[i*N+i] += D_diag_vector[i];

#ifdef HAVE_ARPACK
	// using ARPACK DS{E,A}UPD
	int eigenproblem_status = 0;
	float64_t* eigenvalues_vector = SG_MALLOC(float64_t,m_target_dim+1);
	arpack_dsxupd(W_matrix.matrix,D_diag_vector,true,N,m_target_dim+1,"LA",true,3,false,-1e-9,0.0,
	              eigenvalues_vector,W_matrix.matrix,eigenproblem_status);

	if (eigenproblem_status!=0) 
		SG_ERROR("DSXUPD failed with code %d\n",eigenproblem_status);
	
	SG_FREE(eigenvalues_vector);
#else /* HAVE_ARPACK */
	// using LAPACK DSYGVX
	// requires 2x memory because of dense rhs matrix usage
	int eigenproblem_status = 0;
	float64_t* eigenvalues_vector = SG_MALLOC(float64_t,N);
	float64_t* rhs = SG_CALLOC(float64_t,N*N);
	// fill rhs with diag (for safety reasons zeros will be replaced with 1e-3)
	for (i=0; i<N; i++)
		rhs[i*N+i] = D_diag_vector[i];

	wrap_dsygvx(1,'V','U',N,W_matrix.matrix,N,rhs,N,1,m_target_dim+2,eigenvalues_vector,W_matrix.matrix,&eigenproblem_status);
	
	if (eigenproblem_status)
		SG_ERROR("DSYGVX failed with code: %d.\n",eigenproblem_status);

	SG_FREE(rhs);
	SG_FREE(eigenvalues_vector);

#endif /* HAVE_ARPACK */
	SG_FREE(D_diag_vector);

	SGMatrix<float64_t> new_features = SGMatrix<float64_t>(m_target_dim,N);
	// fill features according to used solver
	for (i=0; i<m_target_dim; i++)
	{
		for (j=0; j<N; j++)
		{
			#ifdef HAVE_ARPACK
				new_features[j*m_target_dim+i] = W_matrix[j*(m_target_dim+1)+i+1];
			#else
				new_features[j*m_target_dim+i] = W_matrix[(i+1)*N+j];
			#endif
		}
	}
}

#endif /* HAVE_LAPACK */
