#define NRANSI
#include "nrutil.h"
#define TOL 2.0e-3

int ncom;
REAL *pcom,*xicom,*xt ;

double linmin(REAL p[], REAL xi[], int n)
{
	REAL brent(REAL ax, REAL bx, REAL cx,
		REAL (*f)(REAL), REAL tol, REAL *xmin);
	REAL f1dim(REAL x);
	void mnbrak(REAL *ax, REAL *bx, REAL *cx, REAL *fa, REAL *fb,
		REAL *fc, REAL (*func)(REAL));
	REAL xx,xmin,fx,fb,fa,bx,ax;

	ncom=n;
	pcom=p ;
	xicom=xi ;
	xt=new REAL[n] ;

	ax=0.0;
	xx=1;
	mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
	brent(ax,xx,bx,f1dim,TOL,&xmin);
	/*	for (j=1;j<=n;j++) {
		xi[j] *= xmin;
		p[j] += xi[j];
	}*/
	delete[] xt ;
	return xmin ;
}
#undef TOL
#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software .). */
