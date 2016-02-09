#include <shogun/mathematics/ajd/FFDiag.h>

#ifdef HAVE_EIGEN3

#include <shogun/base/init.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

void getW(float64_t *C, int *ptN, int *ptK, float64_t *W);

SGMatrix<float64_t> CFFDiag::diagonalize(SGNDArray<float64_t> C0, SGMatrix<float64_t> V0,
						double eps, int itermax)
{
	int n = C0.dims[0];
	int K = C0.dims[2];

	index_t * C_dims = SG_MALLOC(index_t, 3);
	C_dims[0] = C0.dims[0];
	C_dims[1] = C0.dims[1];
	C_dims[2] = C0.dims[2];
	SGNDArray<float64_t> C(C_dims,3);
	memcpy(C.array, C0.array, C0.dims[0]*C0.dims[1]*C0.dims[2]*sizeof(float64_t));

	SGMatrix<float64_t> V;
	if (V0.num_rows == n && V0.num_cols == n)
		V = V0.clone();
	else
		V = SGMatrix<float64_t>::create_identity_matrix(n,1);

	MatrixXd Id(n,n); Id.setIdentity();
	Map<MatrixXd> EV(V.matrix,n,n);

	float64_t inum = 0;
	float64_t df = 1;
	std::vector<float64_t> crit;
	while (df > eps && inum < itermax)
	{
		MatrixXd W = MatrixXd::Zero(n,n);

		getW(C.get_matrix(0),
			 &n, &K,
			 W.data());

		W.transposeInPlace();
		int e = CMath::ceil(log2(W.array().abs().rowwise().sum().maxCoeff()));
		int s = std::max(0,e-1);
		W /= pow(2,s);

		EV = (Id+W) * EV;
		MatrixXd d = MatrixXd::Zero(EV.rows(),EV.cols());
		d.diagonal() = VectorXd::Ones(EV.diagonalSize()).cwiseQuotient((EV * EV.transpose()).diagonal().cwiseSqrt());
		EV = d * EV;

		for (int i = 0; i < K; i++)
		{
			Map<MatrixXd> Ci(C.get_matrix(i), n, n);
			Map<MatrixXd> C0i(C0.get_matrix(i), n, n);
			Ci = EV * C0i * EV.transpose();
		}

		float64_t f = 0;
		for (int i = 0; i < K; i++)
		{
			Map<MatrixXd> C0i(C0.get_matrix(i), n, n);
			MatrixXd F = EV * C0i * EV.transpose();
			f += (F.transpose() * F).diagonal().sum() - F.array().pow(2).matrix().diagonal().sum();
		}

		crit.push_back(f);

		if (inum > 1)
			df = CMath::abs(crit[inum-1]-crit[inum]);

		inum++;
	}

	if (inum == itermax)
		SG_SERROR("Convergence not reached\n")

	return V;

}

void getW(float64_t *C, int *ptN, int *ptK, float64_t *W)
{
	int N=*ptN;
	int K=*ptK;
	int auxij,auxji,auxii,auxjj;
	float64_t z[N][N];
	float64_t y[N][N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			z[i][j] = 0;
			y[i][j] = 0;
		}
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < K; k++)
			{
				auxij = N*N*k+N*i+j;
				auxji = N*N*k+N*j+i;
				auxii = N*N*k+N*i+i;
				auxjj = N*N*k+N*j+j;
				z[i][j] += C[auxii]*C[auxjj];
				y[i][j] += 0.5*C[auxjj]*(C[auxij]+C[auxji]);
			}
		}
	}

	for (int i = 0; i < N-1; i++)
	{
		for (int j = i+1; j < N; j++)
		{
			auxij = N*i+j;
			auxji = N*j+i;
			W[auxij] = (z[j][i]*y[j][i] - z[i][i]*y[i][j])/(z[j][j]*z[i][i]-z[i][j]*z[i][j]);
			W[auxji] = (z[i][j]*y[i][j] - z[j][j]*y[j][i])/(z[j][j]*z[i][i]-z[i][j]*z[i][j]);
		}
	}

	return;
}
#endif //HAVE_EIGEN3
