#include <shogun/mathematics/ajd/JADiagOrth.h>


#include <shogun/base/init.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

float64_t givens_stack(float64_t *A, int M, int K, int p, int q);
void left_rot_stack(float64_t *A, int M, int N, int K, int p, int q, float64_t c, float64_t s);
void right_rot_stack(float64_t *A, int M, int N, int K, int p, int q, float64_t c, float64_t s);
void left_rot_simple(float64_t *A, int m, int n, int p, int q, float64_t c, float64_t s);

SGMatrix<float64_t> CJADiagOrth::diagonalize(SGNDArray<float64_t> C, SGMatrix<float64_t> V0,
						double eps, int itermax)
{
	int m = C.dims[0];
	int L = C.dims[2];

	SGMatrix<float64_t> V;
	if (V0.num_rows == m && V0.num_cols == m)
		V = V0.clone();
	else
		V = SGMatrix<float64_t>::create_identity_matrix(m,1);

	bool more = true;
	int rots = 0;

	while (more)
	{
		more = false;

		for (int p = 0; p < m; p++)
		{
			for (int q = p+1; q < m; q++)
			{
				// computation of Givens angle
				float64_t theta = givens_stack(C.array, m, L, p, q);

				// Givens update
				if (fabs(theta) > eps)
				{
					float64_t c = cos(theta);
					float64_t s = sin(theta);
					left_rot_stack (C.array, m, m, L, p, q, c, s);
					right_rot_stack(C.array, m, m, L, p, q, c, s);
					left_rot_simple(V.matrix, m, m, p, q, c, s);
					rots++;
					more = true;
				}
			}
		}
	}

    return V;
}

/* Givens angle for the pair (p,q) of a stack of K M*M matrices */
float64_t givens_stack(float64_t *A, int M, int K, int p, int q)
{
	int k;
	float64_t diff_on, sum_off, ton, toff;
	float64_t *cm; // A cumulant matrix
	float64_t G11 = 0.0;
	float64_t G12 = 0.0;
	float64_t G22 = 0.0;

	int M2 = M*M;
	int pp = p+p*M;
	int pq = p+q*M;
	int qp = q+p*M;
	int qq = q+q*M;

	for (k=0, cm=A; k<K; k++, cm+=M2)
	{
		diff_on = cm[pp] - cm[qq];
		sum_off = cm[pq] + cm[qp];

		G11 += diff_on * diff_on;
		G22 += sum_off * sum_off;
		G12 += diff_on * sum_off;
	}

	ton  = G11 - G22;
	toff = 2.0 * G12;

	return -0.5 * CMath::atan2 (toff, ton+sqrt(ton*ton+toff*toff));
}

/*
   Ak(mxn) --> R * Ak(mxn) where R rotates the (p,q) rows R =[ c -s ; s c ]
   and Ak is the k-th matrix in the stack
*/
void left_rot_stack(float64_t *A, int M, int N, int K, int p, int q, float64_t c, float64_t s )
{
	int k, ix, iy, cpt;
	int MN = M*N;
	int kMN;
	float64_t nx, ny;

	for (k=0, kMN=0; k<K; k++, kMN+=MN)
	{
		for (cpt=0, ix=p+kMN, iy=q+kMN; cpt<N; cpt++, ix+=M, iy+=M)
		{
			nx = A[ix];
			ny = A[iy];
			A[ix] = c*nx - s*ny;
			A[iy] = s*nx + c*ny;
		}
	}
}

/* Ak(mxn) --> Ak(mxn) x R where R rotates the (p,q) columns R =[ c s ; -s c ]
   and Ak is the k-th M*N matrix in the stack */
void right_rot_stack(float64_t *A, int M, int N, int K, int p, int q, float64_t c, float64_t s )
{
	int k, ix, iy, cpt, kMN;
	int pM = p*M;
	int qM = q*M;
	float64_t nx, ny;

	for (k=0, kMN=0; k<K; k++, kMN+=M*N)
	{
		for (cpt=0, ix=pM+kMN, iy=qM+kMN; cpt<M; cpt++)
		{
			nx = A[ix];
			ny = A[iy];
			A[ix++] = c*nx - s*ny;
			A[iy++] = s*nx + c*ny;
		}
	}
}

/*
   A(mxn) --> R * A(mxn) where R=[ c -s ; s c ]   rotates the (p,q) rows of R
*/
void left_rot_simple(float64_t *A, int m, int n, int p, int q, float64_t c, float64_t s)
{
	int ix = p;
	int iy = q;
	float64_t nx, ny;

	for (int j = 0; j < n; j++, ix+=m, iy+=m)
	{
		nx = A[ix];
		ny = A[iy];
		A[ix] = c*nx - s*ny;
		A[iy] = s*nx + c*ny;
	}
}
