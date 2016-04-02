double algdiv ( double *a, double *b );
double alnrel ( double *a );
double apser ( double *a, double *b, double *x, double *eps );
double bcorr ( double *a0, double *b0 );
double beta ( double a, double b );
double beta_asym ( double *a, double *b, double *lambda, double *eps );
double beta_frac ( double *a, double *b, double *x, double *y, double *lambda,
  double *eps );
void beta_grat ( double *a, double *b, double *x, double *y, double *w,
  double *eps,int *ierr );
void beta_inc ( double *a, double *b, double *x, double *y, double *w,
  double *w1, int *ierr );
void beta_inc_values ( int *n_data, double *a, double *b, double *x, double *fx );
double beta_log ( double *a0, double *b0 );
double beta_pser ( double *a, double *b, double *x, double *eps );
double beta_rcomp ( double *a, double *b, double *x, double *y );
double beta_rcomp1 ( int *mu, double *a, double *b, double *x, double *y );
double beta_up ( double *a, double *b, double *x, double *y, int *n, double *eps );
void binomial_cdf_values ( int *n_data, int *a, double *b, int *x, double *fx );
void cdfbet ( int *which, double *p, double *q, double *x, double *y,
  double *a, double *b, int *status, double *bound );
void cdfbin ( int *which, double *p, double *q, double *s, double *xn,
  double *pr, double *ompr, int *status, double *bound );
void cdfchi ( int *which, double *p, double *q, double *x, double *df,
  int *status, double *bound );
void cdfchn ( int *which, double *p, double *q, double *x, double *df,
  double *pnonc, int *status, double *bound );
void cdff ( int *which, double *p, double *q, double *f, double *dfn,
  double *dfd, int *status, double *bound );
void cdffnc ( int *which, double *p, double *q, double *f, double *dfn,
  double *dfd, double *phonc, int *status, double *bound );
void cdfgam ( int *which, double *p, double *q, double *x, double *shape,
  double *scale, int *status, double *bound );
void cdfnbn ( int *which, double *p, double *q, double *s, double *xn,
  double *pr, double *ompr, int *status, double *bound );
void cdfnor ( int *which, double *p, double *q, double *x, double *mean,
  double *sd, int *status, double *bound );
void cdfpoi ( int *which, double *p, double *q, double *s, double *xlam,
  int *status, double *bound );
void cdft ( int *which, double *p, double *q, double *t, double *df,
  int *status, double *bound );
void chi_noncentral_cdf_values ( int *n_data, double *x, double *lambda,
  int *df, double *cdf );
void chi_square_cdf_values ( int *n_data, int *a, double *x, double *fx );
void cumbet ( double *x, double *y, double *a, double *b, double *cum,
  double *ccum );
void cumbin ( double *s, double *xn, double *pr, double *ompr,
  double *cum, double *ccum );
void cumchi ( double *x, double *df, double *cum, double *ccum );
void cumchn ( double *x, double *df, double *pnonc, double *cum,
  double *ccum );
void cumf ( double *f, double *dfn, double *dfd, double *cum, double *ccum );
void cumfnc ( double *f, double *dfn, double *dfd, double *pnonc,
  double *cum, double *ccum );
void cumgam ( double *x, double *a, double *cum, double *ccum );
void cumnbn ( double *s, double *xn, double *pr, double *ompr,
  double *cum, double *ccum );
void cumnor ( double *arg, double *result, double *ccum );
void cumpoi ( double *s, double *xlam, double *cum, double *ccum );
void cumt ( double *t, double *df, double *cum, double *ccum );
double dbetrm ( double *a, double *b );
double dexpm1 ( double *x );
double dinvnr ( double *p, double *q );
void dinvr ( int *status, double *x, double *fx,
  unsigned long *qleft, unsigned long *qhi );
double dlanor ( double *x );
double dpmpar ( int *i );
void dstinv ( double *zsmall, double *zbig, double *zabsst,
  double *zrelst, double *zstpmu, double *zabsto, double *zrelto );
double dstrem ( double *z );
void dstzr ( double *zxlo, double *zxhi, double *zabstl, double *zreltl );
double dt1 ( double *p, double *q, double *df );
void dzror ( int *status, double *x, double *fx, double *xlo,
  double *xhi, unsigned long *qleft, unsigned long *qhi );
static void E0000 ( int IENTRY, int *status, double *x, double *fx,
  unsigned long *qleft, unsigned long *qhi, double *zabsst,
  double *zabsto, double *zbig, double *zrelst,
  double *zrelto, double *zsmall, double *zstpmu );
static void E0001 ( int IENTRY, int *status, double *x, double *fx,
  double *xlo, double *xhi, unsigned long *qleft,
  unsigned long *qhi, double *zabstl, double *zreltl,
  double *zxhi, double *zxlo );
void erf_values ( int *n_data, double *x, double *fx );
double error_f ( double *x );
double error_fc ( int *ind, double *x );
double esum ( int *mu, double *x );
double eval_pol ( double a[], int *n, double *x );
double exparg ( int *l );
void f_cdf_values ( int *n_data, int *a, int *b, double *x, double *fx );
void f_noncentral_cdf_values ( int *n_data, int *a, int *b, double *lambda,
  double *x, double *fx );
double fifdint ( double a );
double fifdmax1 ( double a, double b );
double fifdmin1 ( double a, double b );
double fifdsign ( double mag, double sign );
long fifidint ( double a );
long fifmod ( long a, long b );
double fpser ( double *a, double *b, double *x, double *eps );
void ftnstop ( const char *msg );
double gam1 ( double *a );
void gamma_inc ( double *a, double *x, double *ans, double *qans, int *ind );
void gamma_inc_inv ( double *a, double *x, double *x0, double *p, double *q,
  int *ierr );
void gamma_inc_values ( int *n_data, double *a, double *x, double *fx );
double gamma_ln1 ( double *a );
double gamma_log ( double *a );
void gamma_rat1 ( double *a, double *x, double *r, double *p, double *q,
  double *eps );
void gamma_values ( int *n_data, double *x, double *fx );
double gamma_x ( double *a );
double gsumln ( double *a, double *b );
int ipmpar ( int *i );
void negative_binomial_cdf_values ( int *n_data, int *f, int *s, double *p,
  double *cdf );
void normal_cdf_values ( int *n_data, double *x, double *fx );
void poisson_cdf_values ( int *n_data, double *a, int *x, double *fx );
double psi ( double *xx );
void psi_values ( int *n_data, double *x, double *fx );
double rcomp ( double *a, double *x );
double rexp ( double *x );
double rlog ( double *x );
double rlog1 ( double *x );
void student_cdf_values ( int *n_data, int *a, double *x, double *fx );
double stvaln ( double *p );
void timestamp ( void );
