#ifdef HAVE_EIGEN3

#include <shogun/mathematics/ajd/JADiagOrth.h>

#include <shogun/base/init.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;
typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;

using namespace shogun;

double givens_stack(double *A, int M, int K, int p, int q);
void LeftRotStack(double *A, int M, int N, int K, int p, int q, double c, double s);
void RightRotStack(double *A, int M, int N, int K, int p, int q, double c, double s);
void LeftRotSimple(double *A, int m, int n, int p, int q, double c, double s);

SGMatrix<float64_t> CJADiagOrth::diagonalize(SGNDArray<float64_t> &C, SGMatrix<float64_t> V0,
						double eps, int itermax)
{
	int m = C.dims[0];
	int L = C.dims[2];	

	SGMatrix<float64_t> V;
	if (V0.num_rows != 0)
	{
		V = V0.clone();
	}
	else
	{					
		V = SGMatrix<float64_t>::create_identity_matrix(m,1);
	}

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
                double theta = givens_stack(C.array, m, L, p, q);

                // Givens update
                if (fabs(theta) > eps)
                {  
                    double c = cos(theta);
                    double s = sin(theta);
                    LeftRotStack (C.array, m, m, L, p, q, c, s);  
	                RightRotStack(C.array, m, m, L, p, q, c, s);  
	                LeftRotSimple(V.matrix, m, m, p, q, c, s);
                    rots++;
                    more = true;
                }
            }
        } 
	}
	
    return V;
}

/* Givens angle for the pair (p,q) of a stack of K M*M matrices */
double givens_stack(double *A, int M, int K, int p, int q)
{
    int k;
    double diff_on, sum_off, ton, toff;
    double *cm; // A cumulant matrix
    double G11 = 0.0;
    double G12 = 0.0;
    double G22 = 0.0;

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

    return -0.5 * atan2 ( toff , ton+sqrt(ton*ton+toff*toff) );
}

/* 
   Ak(mxn) --> R * Ak(mxn) where R rotates the (p,q) rows R =[ c -s ; s c ]  
   and Ak is the k-th matrix in the stack
*/
void LeftRotStack(double *A, int M, int N, int K, int p, int q, double c, double s )
{
    int k, ix, iy, cpt;
    int MN = M*N;
    int kMN;
    double nx, ny;

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
void RightRotStack(double *A, int M, int N, int K, int p, int q, double c, double s ) 
{ 
    int k, ix, iy, cpt, kMN; 
    int pM = p*M;
    int qM = q*M;
    double nx, ny; 

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
void LeftRotSimple(double *A, int m, int n, int p, int q, double c, double s)
{
    int ix = p;
    int iy = q;
    double nx, ny;
    int j;

    for (j=0; j<n; j++, ix+=m, iy+=m) 
    {
        nx = A[ix];
        ny = A[iy];
        A[ix] = c*nx - s*ny;
        A[iy] = s*nx + c*ny;
    }
}
#endif //HAVE_EIGEN3
