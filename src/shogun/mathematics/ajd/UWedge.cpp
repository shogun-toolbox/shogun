#ifdef HAVE_EIGEN3

#include <mathematics/ajd/UWedge.h>

#include <base/init.h>

#include <mathematics/Math.h>
#include <mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

SGMatrix<float64_t> CUWedge::diagonalize(SGNDArray<float64_t> C, SGMatrix<float64_t> V0,
					double eps, int itermax)
{
	int d = C.dims[0];
	int L = C.dims[2];

	SGMatrix<float64_t> V;
	if (V0.num_rows == d && V0.num_cols == d)
	{
		V = V0.clone();
	}
	else
	{
		Map<MatrixXd> C0(C.get_matrix(0),d,d);
		EigenSolver<MatrixXd> eig;
		eig.compute(C0);

		// sort eigenvectors
		MatrixXd eigenvectors = eig.pseudoEigenvectors();
		MatrixXd eigenvalues = eig.pseudoEigenvalueMatrix();

		bool swap = false;
		do
		{
			swap = false;
			for (int j = 1; j < d; j++)
			{
				if ( eigenvalues(j,j) > eigenvalues(j-1,j-1) )
				{
					std::swap(eigenvalues(j,j),eigenvalues(j-1,j-1));
					eigenvectors.col(j).swap(eigenvectors.col(j-1));
					swap = true;
				}
			}

		} while(swap);

		V = SGMatrix<float64_t>::create_identity_matrix(d,1);
		Map<MatrixXd> EV(V.matrix, d,d);
		EV = eigenvalues.cwiseAbs().cwiseSqrt().inverse() * eigenvectors.transpose();
	}
	Map<MatrixXd> EV(V.matrix, d,d);

	index_t * Cs_dims = SG_MALLOC(index_t, 3);
	Cs_dims[0] = d;
	Cs_dims[1] = d;
	Cs_dims[2] = L;
	SGNDArray<float64_t> Cs(Cs_dims,3);
	memcpy(Cs.array, C.array, Cs.dims[0]*Cs.dims[1]*Cs.dims[2]*sizeof(float64_t));

	MatrixXd Rs(d,L);
	std::vector<float64_t> crit;
	crit.push_back(0.0);
	for (int l = 0; l < L; l++)
	{
		Map<MatrixXd> Ci(C.get_matrix(l),d,d);
		Map<MatrixXd> Csi(Cs.get_matrix(l),d,d);
		Ci = 0.5 * (Ci + Ci.transpose());
		Csi = EV * Ci * EV.transpose();
		Rs.col(l) = Csi.diagonal();
		crit.back() += Csi.cwiseAbs2().sum() - Rs.col(l).cwiseAbs2().sum();
	}

	float64_t iter = 0;
	float64_t improve = 10;
	while (improve > eps && iter < itermax)
	{
		MatrixXd B = Rs * Rs.transpose();

		MatrixXd C1 = MatrixXd::Zero(d,d);
		for (int id = 0; id < d; id++)
		{
			// rowSums
			for (int l = 0; l < L; l++)
			{
				Map<MatrixXd> Csi(Cs.get_matrix(l),d,d);
				C1.row(id) += Csi.row(id) * Rs(id,l);
			}
		}

		MatrixXd D0 = B.cwiseProduct(B.transpose()) - B.diagonal() * B.diagonal().transpose();
		MatrixXd A0 = MatrixXd::Identity(d,d) + (C1.cwiseProduct(B) - B.diagonal().asDiagonal() * C1.transpose()).cwiseQuotient(D0+MatrixXd::Identity(d,d));
		EV = A0.inverse() * EV;

		Map<MatrixXd> C0(C.get_matrix(0),d,d);
		MatrixXd Raux = EV * C0 * EV.transpose();
		MatrixXd aux = Raux.diagonal().cwiseAbs().cwiseSqrt().asDiagonal().inverse();
		EV = aux * EV;

		crit.push_back(0.0);
		for (int l = 0; l < L; l++)
		{
			Map<MatrixXd> Ci(C.get_matrix(l),d,d);
			Map<MatrixXd> Csi(Cs.get_matrix(l),d,d);
			Csi = EV * Ci * EV.transpose();
			Rs.col(l) = Csi.diagonal();
			crit.back() += Csi.cwiseAbs2().sum() - Rs.col(l).cwiseAbs2().sum();
		}

		improve = CMath::abs(crit.back() - crit[iter]);
		iter++;
	}

	if (iter == itermax)
		SG_SERROR("Convergence not reached\n")

	return V;

}

#endif //HAVE_EIGEN3
