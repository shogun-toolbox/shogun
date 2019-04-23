/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/converter/ica/Jade.h>

#include <shogun/features/DenseFeatures.h>


#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/ajd/JADiagOrth.h>

#ifdef DEBUG_JADE
#include <iostream>
#endif

using namespace shogun;
using namespace Eigen;

Jade::Jade() : ICAConverter()
{
	init();
}

void Jade::init()
{
	m_cumulant_matrix = SGMatrix<float64_t>();
	SG_ADD(&m_cumulant_matrix, "cumulant_matrix", "m_cumulant_matrix");
}

Jade::~Jade()
{
}

SGMatrix<float64_t> Jade::get_cumulant_matrix() const
{
	return m_cumulant_matrix;
}

void Jade::fit_dense(std::shared_ptr<DenseFeatures<float64_t>> features)
{
	ASSERT(features);

	auto X = features->get_feature_matrix();

	int n = X.num_rows;
	int T = X.num_cols;
	int m = n;

	Eigen::Map<MatrixXd> EX(X.matrix,n,T);

	// Mean center X
	VectorXd mean = (EX.rowwise().sum() / (float64_t)T);
	MatrixXd SPX = EX.colwise() - mean;

	MatrixXd cov = (SPX * SPX.transpose()) / (float64_t)T;

	#ifdef DEBUG_JADE
	std::cout << "cov" << std::endl;
	std::cout << cov << std::endl;
	#endif

	// Whitening & Projection onto signal subspace
	SelfAdjointEigenSolver<MatrixXd> eig;
	eig.compute(cov);

	#ifdef DEBUG_JADE
	std::cout << "eigenvectors" << std::endl;
	std::cout << eig.eigenvectors() << std::endl;

	std::cout << "eigenvalues" << std::endl;
	std::cout << eig.eigenvalues().asDiagonal() << std::endl;
	#endif

	// Scaling
	VectorXd scales = eig.eigenvalues().cwiseSqrt();
	MatrixXd B = scales.cwiseInverse().asDiagonal() * eig.eigenvectors().transpose();

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
	Eigen::Map<MatrixXd> CM(m_cumulant_matrix.matrix,m,m*nbcm);
	MatrixXd R(m,m); R.setIdentity();
	MatrixXd Qij = MatrixXd::Zero(m,m); // Temp for a cum. matrix
	VectorXd Xim = VectorXd::Zero(m); // Temp
	VectorXd Xjm = VectorXd::Zero(m); // Temp
	VectorXd Xijm = VectorXd::Zero(m); // Temp
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
		Eigen::Map<MatrixXd> EM(M.get_matrix(i),m,m);
		EM = CM.block(0,i*m,m,m);
	}

	// Diagonalize
	SGMatrix<float64_t> Q = JADiagOrth::diagonalize(M, m_mixing_matrix, tol, max_iter);
	Eigen::Map<MatrixXd> EQ(Q.matrix,m,m);
	EQ = -1 * EQ.inverse();

	#ifdef DEBUG_JADE
	std::cout << "diagonalizer" << std::endl;
	std::cout << EQ << std::endl;
	#endif

	// Separating matrix
	SGMatrix<float64_t> sep_matrix = SGMatrix<float64_t>(m,m);
	Eigen::Map<MatrixXd> C(sep_matrix.matrix,m,m);
	C = EQ.transpose() * B;

	// Sort
	VectorXd A = (B.inverse()*EQ).cwiseAbs2().colwise().sum();
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
	VectorXd signs = VectorXd::Zero(m);
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

	m_mixing_matrix = SGMatrix<float64_t>(m,m);
	Eigen::Map<MatrixXd> Emixing_matrix(m_mixing_matrix.matrix,m,m);
	Emixing_matrix = C.inverse();
}
