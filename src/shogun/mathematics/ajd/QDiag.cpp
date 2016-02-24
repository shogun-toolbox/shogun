#include <shogun/mathematics/ajd/QDiag.h>

#ifdef HAVE_EIGEN3

#include <shogun/base/init.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

SGMatrix<float64_t> CQDiag::diagonalize(SGNDArray<float64_t> C, SGMatrix<float64_t> V0,
						double eps, int itermax)
{
	int N = C.dims[0];
	int T = C.dims[2];

	SGMatrix<float64_t> V;
	if (V0.num_rows == N && V0.num_cols == N)
	{
		V = V0.clone();
	}
	else
	{
		V = SGMatrix<float64_t>(N,N);

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				V(i,j) = CMath::randn_double();
		}
	}

	std::vector<float64_t> p(T,1.0/T);

	MatrixXd C0 = MatrixXd::Identity(N,N);

	Map<MatrixXd> EV(V.matrix,N,N);
	EV = EV * VectorXd::Ones(EV.rows()).cwiseQuotient((EV.transpose() * C0 * EV).cwiseSqrt().diagonal()).asDiagonal();

	MatrixXd P = MatrixXd::Zero(N,N);
	for (int i = 0; i < N; i++)
		P(i,N-1-i) = 1;

	std::vector<bool> issymmetric(T);
	for (int l = 0; l < T; l++)
	{
		Map<MatrixXd> Ci(C.get_matrix(l),N,N);

		Ci = P * Ci * P.transpose();

		if ( (Ci - Ci.transpose()).sum() > 1e-6 )
			issymmetric[l] = false;
		else
			issymmetric[l] = true;
	}

	C0 = P * C0 * P.transpose();
	EV = P * EV;

	// initialisations for OKN^3
	MatrixXd D = MatrixXd::Zero(N,N);
	for (int t = 0; t < T; t++)
	{
		Map<MatrixXd> Ci(C.get_matrix(t),N,N);
		MatrixXd M1 = Ci * EV;

		if (issymmetric[t])
		{
			D = D + 2*p[t] * M1 * M1.transpose();
		}
		else
		{
			MatrixXd M2 = Ci.transpose() * EV;
			D = D + p[t] * (M1*M1.transpose() + M2*M2.transpose());
		}
	}

	int iter = 0;
	float64_t deltacrit = 1.0;
	std::vector<float64_t> crit;
	while ( iter < itermax && deltacrit > eps )
	{
		float64_t delta_w = 0.0;

		for (int i = 0; i < N; i++)
		{
			VectorXd w = EV.col(i);

			for (int t = 0; t < T; t++)
			{
				Map<MatrixXd> Ci(C.get_matrix(t),N,N);
				VectorXd m1 = Ci * w;

				if (issymmetric[t])
				{
					D = D - 2*p[t] * m1 * m1.transpose();
				}
				else
				{
					VectorXd m2 = Ci.transpose() * w;
					D = D - p[t] * (m1*m1.transpose() + m2*m2.transpose());
				}
			}

			EigenSolver<MatrixXd> eig;
			eig.compute(D);

			// sort eigenvectors
			MatrixXd eigenvectors = eig.pseudoEigenvectors();
			VectorXd eigenvalues = eig.pseudoEigenvalueMatrix().diagonal();

			bool swap = false;
			do
			{
				swap = false;
				for (int j = 1; j < D.rows(); j++)
				{
					if( eigenvalues[j] > eigenvalues[j-1] )
					{
						std::swap(eigenvalues[j],eigenvalues[j-1]);
						eigenvectors.col(j).swap(eigenvectors.col(j-1));
						swap = true;
					}
				}

			} while(swap);

			VectorXd w_new = eigenvectors.col(N-1);
			delta_w = std::max(delta_w, std::min(sqrt((w-w_new).cwiseAbs2().sum()), sqrt((w+w_new).cwiseAbs2().sum())));

			for (int t = 0; t < T; t++)
			{
				Map<MatrixXd> Ci(C.get_matrix(t),N,N);

				VectorXd m1 = Ci * w_new;
				if (issymmetric[t])
				{
					D = D + 2*p[t] * m1 * m1.transpose();
				}
				else
				{
					VectorXd m2 = Ci.transpose() * w_new;
					D = D + p[t] * (m1*m1.transpose() + m2*m2.transpose());
				}
			}
			EV.col(i) = w_new;
		}

		// err
		crit.push_back(0.0);
		EV = EV * (EV.transpose() * C0 * EV).diagonal().cwiseSqrt().asDiagonal().inverse();
		for (int t = 0; t < T; t++)
		{
			Map<MatrixXd> Ci(C.get_matrix(t),N,N);
			MatrixXd eD = EV.transpose() * Ci * EV;
			eD.diagonal() = VectorXd::Zero(eD.rows());
			crit.back() = crit.back() + p[t]*eD.cwiseAbs2().sum();
		}
		crit.back() = crit.back() / (N*N - N);

		if (iter > 1)
			deltacrit = CMath::abs( crit[iter] - crit[iter-1] );

		iter++;
	}

	EV = (P.transpose() * EV).transpose();

	if (iter == itermax)
		SG_SERROR("Convergence not reached\n")

	return V;
}
#endif //HAVE_EIGEN3
