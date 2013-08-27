/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 */

#include <shogun/converter/ica/Jade.h>

#include <shogun/features/DenseFeatures.h>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/ajd/JADiagOrth.h>

#ifdef DEBUG_JADE
#include <iostream>
#endif

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;
typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;

using namespace shogun;

CJade::CJade() : CICAConverter()
{	
	init();
}

void CJade::init()
{
	m_cumulant_matrix = SGMatrix<float64_t>();
	SG_ADD(&m_cumulant_matrix, "cumulant_matrix", "m_cumulant_matrix", MS_NOT_AVAILABLE);
}

CJade::~CJade()
{
}

SGMatrix<float64_t> CJade::get_cumulant_matrix() const
{
	return m_cumulant_matrix;
}

CFeatures* CJade::apply(CFeatures* features)
{
	ASSERT(features);	
	SG_REF(features);

	SGMatrix<float64_t> X = ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	int n = X.num_rows;
	int T = X.num_cols;
	int m = n;

	Eigen::Map<EMatrix> EX(X.matrix,n,T);

	// Mean center X
	EVector mean = (EX.rowwise().sum() / (float64_t)T);	
	EMatrix SPX = EX.colwise() - mean;

	EMatrix cov = (SPX * SPX.transpose()) / (float64_t)T;

	#ifdef DEBUG_JADE
	std::cout << "cov" << std::endl;
	std::cout << cov << std::endl;
	#endif

	// Whitening & Projection onto signal subspace
	SelfAdjointEigenSolver<EMatrix> eig;
	eig.compute(cov);

	#ifdef DEBUG_JADE
	std::cout << "eigenvectors" << std::endl;
	std::cout << eig.eigenvectors() << std::endl;
	
	std::cout << "eigenvalues" << std::endl;	
	std::cout << eig.eigenvalues().asDiagonal() << std::endl;
	#endif

	// Scaling
	EVector scales = eig.eigenvalues().cwiseSqrt();
	EMatrix B = scales.cwiseInverse().asDiagonal() * eig.eigenvectors().transpose();

	#ifdef DEBUG_JADE
	std::cout << "whitener" << std::endl;	
	std::cout << B << std::endl;
	#endif

	// Sphering
	SPX = B * SPX;

	// Estimation of the cumulant matrices
	int dimsymm = (m * ( m + 1)) / 2; // Dim. of the space of real symm matrices
	int nbcm = dimsymm; //  number of cumulant matrices	
	m_cumulant_matrix = SGMatrix<float64_t>(m,m*nbcm);	// Storage for cumulant matrices
	Eigen::Map<EMatrix> CM(m_cumulant_matrix.matrix,m,m*nbcm); 
	EMatrix R(m,m); R.setIdentity();
	EMatrix Qij = EMatrix::Zero(m,m); // Temp for a cum. matrix
	EVector Xim = EVector::Zero(m); // Temp
	EVector Xjm = EVector::Zero(m); // Temp
	EVector Xijm = EVector::Zero(m); // Temp
	int Range = 0;

	for (int im = 0; im < m; im++)
	{ 
		Xim = SPX.row(im);	
		Xijm = Xim.cwiseProduct(Xim);
		Qij = SPX.cwiseProduct(Xijm.replicate(1,m).transpose()) * SPX.transpose() / (float)T - R - 2*R.col(im)*R.col(im).transpose();
		CM.block(0,Range,m,m) = Qij;
		Range = Range + m;	
		for (int jm = 0; jm < im; jm++)
		{
			Xjm = SPX.row(jm);
			Xijm = Xim.cwiseProduct(Xjm);
			Qij = SPX.cwiseProduct(Xijm.replicate(1,m).transpose()) * SPX.transpose() / (float)T - R.col(im)*R.col(jm).transpose() - R.col(jm)*R.col(im).transpose();
			CM.block(0,Range,m,m) =  sqrt(2)*Qij;
			Range = Range + m;	
		}
	}

	#ifdef DEBUG_JADE
	std::cout << "cumulatant matrices" << std::endl;
	std::cout << CM << std::endl;
	#endif

	// Stack cumulant matrix into ND Array
	index_t * M_dims = SG_MALLOC(index_t, 3);
	M_dims[0] = m;
	M_dims[1] = m;
	M_dims[2] = nbcm;
	SGNDArray< float64_t > M(M_dims, 3);
	
	for (int i = 0; i < nbcm; i++)
	{
		Eigen::Map<EMatrix> EM(M.get_matrix(i),m,m);
		EM = CM.block(0,i*m,m,m);
	}
	
	// Diagonalize
	SGMatrix<float64_t> Q = CJADiagOrth::diagonalize(M);
	Eigen::Map<EMatrix> EQ(Q.matrix,m,m);
	EQ = -1 * EQ.inverse();
	
	#ifdef DEBUG_JADE
	std::cout << "diagonalizer" << std::endl;
	std::cout << EQ << std::endl;
	#endif

	// Separating matrix
	SGMatrix<float64_t> sep_matrix = SGMatrix<float64_t>(m,m);
	Eigen::Map<EMatrix> C(sep_matrix.matrix,m,m);
	C = EQ.transpose() * B;

	// Sort
	EVector A = (B.inverse()*EQ).cwiseAbs2().colwise().sum();
	bool swap = false;
	do
	{
		swap = false;
		for (int j = 1; j < n; j++)
		{
			if ( A(j) < A(j-1) )
			{
				std::swap(A(j),A(j-1));
				C.col(j).swap(C.col(j-1));
				swap = true;	
			}						
		}
	
	} while(swap);

	for (int j = 0; j < m/2; j++)
		C.row(j).swap( C.row(m-1-j) ); 

	// Fix Signs
	EVector signs = EVector::Zero(m);
	for (int i = 0; i < m; i++)
	{
		if( C(i,0) < 0 )
			signs(i) = -1;
		else
			signs(i) = 1;
	}
	C = signs.asDiagonal() * C;

	#ifdef DEBUG_JADE
	std::cout << "un mixing matrix" << std::endl;
	std::cout << C << std::endl;
	#endif

	// Unmix
	EX = C * EX;
	
	m_mixing_matrix = SGMatrix<float64_t>(m,m);
	Eigen::Map<EMatrix> Emixing_matrix(m_mixing_matrix.matrix,m,m);
	Emixing_matrix = C.inverse();
	
	return features;
}

#endif // HAVE_EIGEN3
