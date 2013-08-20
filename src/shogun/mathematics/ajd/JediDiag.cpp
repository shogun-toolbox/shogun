#ifdef HAVE_EIGEN3

#include <shogun/mathematics/ajd/JediDiag.h>

#include <shogun/base/init.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;
typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;

using namespace shogun;

void sweepjedi(float64_t *C, int *pMatSize, int *pMatNumber,
				float64_t *s_max, float64_t *sh_max, float64_t *A); 

void iterJDI(float64_t *C, int *pMatSize, int *pMatNumber, int *ptn,int *ptm,
			float64_t *s_max, float64_t *sh_max, float64_t *A);

SGMatrix<float64_t> CJediDiag::diagonalize(SGNDArray<float64_t> C, SGMatrix<float64_t> V0,
						double eps, int itermax)
{
	int d = C.dims[0];
	int L = C.dims[2];

	SGMatrix<float64_t> V;
	if (V0.num_rows != 0)
		V = V0.clone();
	else					
		V = SGMatrix<float64_t>::create_identity_matrix(d,1);

	int iter = 0;
	float64_t sh_max = 1;
	float64_t s_max = 1;
	while (iter < itermax && ( (sh_max>eps) || (s_max>eps) ))
	{
		sh_max = 0;
		s_max = 0;
		sweepjedi(C.array,
					&d, &L,
					&s_max, &sh_max,
					V.matrix);
		iter++;
	}
	
	if (iter == itermax)
		SG_SERROR("Convergence not reached\n")
	
	Eigen::Map<EMatrix> EV(V.matrix,d,d);
	EV = EV.inverse();
	
	return V;
}

void sweepjedi(float64_t *C, int *pMatSize, int *pMatNumber,
				float64_t *s_max, float64_t *sh_max, float64_t *A) 
{
	for (int n=2;n<=*pMatSize;n++) 
		for (int m=1;m<n;m++) 
			iterJDI(C, pMatSize, pMatNumber, &n, &m, s_max, sh_max, A);
			
	int MS=*pMatSize;
	int MN=*pMatNumber;
	float64_t col_norm[MS];
	
	for (int i=0;i<MS;i++) 
	{
		col_norm[i] = 0;
		
		for (int j=0;j<MS;j++)
			col_norm[i] += A[i*MS+j]*A[i*MS+j];

		col_norm[i] = sqrt(col_norm[i]);
	}
	
	float64_t daux=1;
	float64_t d[MS];
	
	for (int i=0;i<MS;i++) 
		daux*=col_norm[i];
	
	daux=pow((float64_t)daux,(float64_t) 1/MS);
	
	for (int i=0;i<MS;i++) 
		d[i]=daux/col_norm[i];
	
	for (int i=0;i<MS;i++) 
		for (int j=0;j<MS;j++)
			A[j*MS+i] *= d[j];
	
	for (int k=0;k<MN;k++) 
	{
		for (int i=0;i<MS;i++) 
		{
			for (int j=0;j<MS;j++) 
				C[k*MS*MS+i*MS+j] *= (1/d[i])*(1/d[j]);
		}
	}
}

void iterJDI(float64_t *C, int *pMatSize, int *pMatNumber, int *ptn,int *ptm,
			float64_t *s_max, float64_t *sh_max, float64_t *A) 
{
	int n=*ptn-1;
	int m=*ptm-1;
	int MN=*pMatNumber;
	int MS=*pMatSize;

	float64_t tmm[MN];
	float64_t tnn[MN];
	float64_t tmn[MN];
	for (int i = 0, d3 = 0; i < MN; i++, d3+=MS*MS) 
	{
		tmm[i] = C[d3+m*MS+m];
		tnn[i] = C[d3+n*MS+n];
		tmn[i] = C[d3+n*MS+m];
	}

	// here we evd
	float64_t G[MN][3];
	float64_t evectors[9], evalues[3];
	for (int i = 0; i < MN; i++) 
	{
		G[i][0] = 0.5*(tmm[i]+tnn[i]);	
		G[i][1] = 0.5*(tmm[i]-tnn[i]);
		G[i][2] = tmn[i];	
	}
	float64_t GG[9];
	for (int i = 0; i < 3; i++) 
	{
		for (int j = 0; j <= i; j++) 
		{
			GG[3*j+i] = 0;
			
			for (int k = 0; k < MN; k++) 
				GG[3*j+i] += G[k][i]*G[k][j];
			
			GG[3*i+j] = GG[3*j+i];
		}
	}
	
	for (int i = 0; i < 3; i++) 
		GG[3*i] *= -1;
	
	Eigen::Map<EMatrix> EGG(GG,3,3);
	
	EigenSolver<EMatrix> eig;
	eig.compute(EGG);
	
	EMatrix eigenvectors = eig.pseudoEigenvectors();
	EVector eigenvalues = eig.pseudoEigenvalueMatrix().diagonal();
	
	eigenvectors.col(0) = eigenvectors.col(0).normalized();
	eigenvectors.col(1) = eigenvectors.col(1).normalized();
	eigenvectors.col(2) = eigenvectors.col(2).normalized();
	
	memcpy(evectors, eigenvectors.data(), 9*sizeof(float64_t));
	memcpy(evalues, eigenvalues.data(), 3*sizeof(float64_t));
	
	float64_t tmp_evec[3],tmp_eval;
	if (fabs(evalues[1])<fabs(evalues[2])) 
	{
		tmp_eval = evalues[1];
		evalues[1] = evalues[2];
		evalues[2] = tmp_eval;
		memcpy(tmp_evec,&evectors[3],3*sizeof(float64_t));
		memcpy(&evectors[3],&evectors[6],3*sizeof(float64_t));
		memcpy(&evectors[6],tmp_evec,3*sizeof(float64_t));
	}
	if (fabs(evalues[0])<fabs(evalues[1])) 
	{
		tmp_eval = evalues[0];
		evalues[0] = evalues[1];
		evalues[1] = tmp_eval;
		memcpy(tmp_evec,evectors,3*sizeof(float64_t));
		memcpy(evectors,&evectors[3],3*sizeof(float64_t));
		memcpy(&evectors[3],tmp_evec,3*sizeof(float64_t));
	}
	if (fabs(evalues[1])<fabs(evalues[2])) 
	{
		tmp_eval = evalues[1];
		evalues[1] = evalues[2];
		evalues[2] = tmp_eval;
		memcpy(tmp_evec,&evectors[3],3*sizeof(float64_t));
		memcpy(&evectors[3],&evectors[6],3*sizeof(float64_t));
		memcpy(&evectors[6],tmp_evec,3*sizeof(float64_t));
	}

	float64_t aux[9];
	float64_t diagaux[3];
	float64_t v[3];
	for (int i = 0; i < 9; i++) 
		aux[i] = evectors[i];
	
	for (int i = 0; i < 3; i++) 
		aux[3*i] *= -1;
	
	for (int i = 0; i < 3; i++) 
	{
		diagaux[i]	= 0;
		
		for (int j = 0; j < 3; j++) 
			diagaux[i] += evectors[3*i+j] * aux[3*i+j];
		
		diagaux[i] = 1/sqrt(fabs(diagaux[i]));
	}
	
	int ind_min=2;
	
	for (int i = 0; i < 3; i++) 
		v[i] = evectors[3*ind_min+i]*diagaux[ind_min];
	
	if (v[2]<0) 
		for (int i = 0; i < 3; i++) 
			v[i]*=-1;
	
	float64_t ch = sqrt( (1+sqrt(1+v[0]*v[0]))/2 );
	float64_t sh = v[0]/(2*ch);
	float64_t c = sqrt( ( 1 + v[2]/sqrt(1+v[0]*v[0]) )/2 );
	float64_t s = -v[1]/( 2*c*sqrt( (1+v[0]*v[0]) ) );
	*s_max = std::max(*s_max,fabs(s));
	*sh_max = std::max(*sh_max,fabs(sh));
	
	float64_t rm11=c*ch - s*sh;
	float64_t rm12=c*sh - s*ch;
	float64_t rm21=c*sh + s*ch;
	float64_t rm22=c*ch + s*sh;

	float64_t h_slice1,h_slice2;
	float64_t buf[MS][MN];
	
	for (int i = 0; i < MS ;i++) 
	{
		for (int j = 0, d3 = 0; j < MN; j++, d3+=MS*MS) 
		{
			h_slice1 = C[d3+i*MS+m];
			h_slice2 = C[d3+i*MS+n];
			buf[i][j] = rm11*h_slice1 + rm21*h_slice2;
			C[d3+i*MS+n] = rm12*h_slice1 + rm22*h_slice2;
			C[d3+i*MS+m] = buf[i][j];
		}
	}

	for (int i = 0; i < MS; i++) 
	{
		for (int j = 0, d3 = 0; j < MN; j++, d3+=MS*MS) 
			C[d3+m*MS+i] = buf[i][j];
	}	
	for (int i = 0; i < MS; i++) 
	{
		for (int j = 0, d3 = 0; j < MN; j++, d3+=MS*MS) 
			C[d3+n*MS+i] = C[d3+i*MS+n];
	}
	for (int i = 0; i < MN; i++) 
	{
		C[i*MS*MS+m*MS+m] = (rm11*rm11)*tmm[i]+(2*rm11*rm21)*tmn[i]+(rm21*rm21)*tnn[i];
		C[i*MS*MS+n*MS+m] = (rm11*rm12)*tmm[i]+(rm11*rm22+rm12*rm21)*tmn[i]+(rm21*rm22)*tnn[i];
		C[i*MS*MS+m*MS+n] = C[i*MS*MS+n*MS+m];
		C[i*MS*MS+n*MS+n] = (rm12*rm12)*tmm[i]+(2*rm12*rm22)*tmn[i]+(rm22*rm22)*tnn[i];
	}

	float64_t col1;
	float64_t col2;
	
	for (int i = 0; i < MS; i++) 
	{
		col1 = A[m*MS+i];
		col2 = A[n*MS+i];
		A[m*MS+i] = rm22*col1 - rm12*col2;
		A[n*MS+i] = -rm21*col1 + rm11*col2;
	}
}
#endif //HAVE_EIGEN3
