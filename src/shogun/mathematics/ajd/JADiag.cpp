#ifdef HAVE_EIGEN3

#include <shogun/mathematics/ajd/JADiag.h>

#include <shogun/base/init.h>
#include <shogun/lib/common.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;
typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;

using namespace shogun;

void jadiagw(double c[], double w[], int *ptn, int *ptm, double a[],
	         double *logdet, double *decr, double *result);

SGMatrix<float64_t> CJADiag::diagonalize(SGNDArray<float64_t> &C, SGMatrix<float64_t> V0,
						double eps, int itermax)
{
	int L = C.dims[2];
	int d = C.dims[0];
	double decr = 1;
	double logdet = log(5.184e17);
	
	SGMatrix<float64_t> V;
	if(V0.num_rows != 0)
	{
		V = V0.clone();
	}
	else
	{	
		V = SGMatrix<float64_t>::create_identity_matrix(d,1);
	}
	
	double result = 0;
	
	EVector w(L);
	w.setOnes();
	
	EMatrix ctot(d, d*L);
	for(int i = 0; i < L; i++)
	{
		Eigen::Map<EMatrix> Ci(C.get_matrix(i),d,d);
		ctot.block(0,i*d,d,d) = Ci;
	}
	
	std::vector<double> crit;
	int iter = 0;
	
	while(decr > eps && iter < itermax)
	{		
		if(logdet == 0)// is NA
		{
			SG_SERROR("log det does not exist\n")
			break;
		}
		
		jadiagw(ctot.data(),
				w.data(),
				&d, &L,
				V.matrix,
				&logdet,
				&decr,
				&result);			
		
		crit.push_back(result);
		iter = iter + 1;
	}
	
	if(iter == itermax)
	{
		SG_SERROR("Convergence not reached\n")
	}
	
	return V;

}

void jadiagw(double c[], double w[], int *ptn, int *ptm, double a[],
		double *logdet, double *decr, double *result)
{
	int n = *ptn;
	int m = *ptm;
	//int	i1,j1;
	int	n2 = n*n, mn2 = m*n2,
	i, ic, ii, ij, j, jc, jj, k, k0;
	double  sumweigh, p2, q1, p, q,
	alpha, beta, gamma, a12, a21, /*tiny,*/ det;
	register double tmp1, tmp2, tmp, weigh;

	for (sumweigh = 0, i = 0; i < m; i++)
	{
		sumweigh += w[i];
	}

	det = 1;
	*decr = 0;
	
	for (i = 1, ic = n; i < n ; i++, ic += n)
	{
		for (j = jc = 0; j < i; j++, jc += n) 
		{
			ii = i + ic;
			jj = j + jc;
			ij = i + jc;
			
			for (q1 = p2 = p = q = 0, k0 = k = 0; k0 < m; k0++, k += n2) 
			{
				weigh = w[k0];
				tmp1 = c[ii+k];
				tmp2 = c[jj+k];
				tmp = c[ij+k];
				p += weigh*tmp/tmp1;
				q += weigh*tmp/tmp2;
				q1 += weigh*tmp1/tmp2;
				p2 += weigh*tmp2/tmp1;
			}
			
			q1 /= sumweigh;
			p2 /= sumweigh;
			p /= sumweigh;
			q /= sumweigh;
			beta = 1 - p2*q1;// p1 = q2 = 1
			
			if (q1 <= p2)// the same as q1*q2 <= p1*p2 
			{			
				alpha = p2*q - p;// q2 = 1 
					
				if (fabs(alpha) - beta < 10e-20)// beta <= 0 always
				{	
					beta = -1;
					gamma = p/p2;
				} 
				else 
				{
					gamma = - (p*beta + alpha)/p2;// p1 = 1 
				}
				
				*decr += sumweigh*(p*p - alpha*alpha/beta)/p2;
			} 
			else 
			{
				gamma = p*q1 - q;// p1 = 1 
				
				if (fabs(gamma) - beta < 10e-20)// beta <= 0 always 
				{	
					beta = -1;
					alpha = q/q1;
				} 
				else
				{
					alpha = - (q*beta + gamma)/q1;// q2 = 1
				}
				
				*decr += sumweigh*(q*q - gamma*gamma/beta)/q1;
			}
			
			tmp = (beta - sqrt(beta*beta - 4*alpha*gamma))/2;
			a12 = gamma/tmp;
			a21 = alpha/tmp;
			
			for (k = 0; k < mn2; k += n2) 
			{
				for (ii = i, jj = j; ii < ij; ii += n, jj += n) 
				{
					tmp = c[ii+k];
					c[ii+k] += a12*c[jj+k];
					c[jj+k] += a21*tmp;
				}// at exit ii = ij = i + jc 
				
				tmp = c[i+ic+k];
				c[i+ic+k] += a12*(2*c[ij+k] + a12*c[jj+k]);
				c[jj+k] += a21*c[ij+k];
				c[ij+k] += a21*tmp;// = element of index j,i
				 
				for (; ii < ic; ii += n, jj++) 
				{
					tmp = c[ii+k];
					c[ii+k] += a12*c[jj+k];
					c[jj+k] += a21*tmp;
				}
				
				for (; ++ii, ++jj < jc+n; ) 
				{
					tmp = c[ii+k];
					c[ii+k] += a12*c[jj+k];
					c[jj+k] += a21*tmp;
				}
				
			}
			
			for (k = 0; k < n2; k += n) 
			{
				tmp = a[i+k];
				a[i+k] += a12*a[j+k];
				a[j+k] += a21*tmp;
			}
			
			det *= 1 - a12*a21;// compute determinant 
		}
	}
	
	*logdet += 2*sumweigh*log(det);
	
	for (tmp = 0, k0 = k = 0; k0 < m; k0++, k += n2) 
	{
		for (det = 1, ii = 0; ii < n2; ii += n+1)
		{
			det *= c[ii+k];
			tmp += w[k0]*log(det);
		}
	}
	
	*result = tmp - *logdet;

	return;
}
#endif //HAVE_EIGEN3
