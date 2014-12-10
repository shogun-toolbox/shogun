#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include <shogun/lib/config.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>

#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/optimization/liblinear/tron.h>

using namespace shogun;

double tron_ddot(const int N, const double *X, const int incX, const double *Y, const int incY)
{
#ifdef HAVE_LAPACK
	return cblas_ddot(N,X,incX,Y,incY);
#else
	double dot = 0.0;
	for (int32_t i=0; i<N; i++)
		dot += X[incX*i]*Y[incY*i];
	return dot;
#endif
}

double tron_dnrm2(const int N, const double *X, const int incX)
{
#ifdef HAVE_LAPACK
	return cblas_dnrm2(N,X,incX);
#else
	double dot = 0.0;
	for (int32_t i=0; i<N; i++)
		dot += X[incX*i]*X[incX*i];
	return sqrt(dot);
#endif
}

void tron_dscal(const int N, const double alpha, double *X, const int incX)
{
#ifdef HAVE_LAPACK
	return cblas_dscal(N,alpha,X,incX);
#else
	for (int32_t i=0; i<N; i++)
		X[i]*= alpha;
#endif
}

void tron_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY)
{
#ifdef HAVE_LAPACK
	cblas_daxpy(N,alpha,X,incX,Y,incY);
#else
	for (int32_t i=0; i<N; i++)
		Y[i] += alpha*X[i];
#endif
}

CTron::CTron(const function *f, float64_t e, int32_t it)
: CSGObject()
{
	this->fun_obj=const_cast<function *>(f);
	this->eps=e;
	this->max_iter=it;
}

CTron::~CTron()
{
}

void CTron::tron(float64_t *w, float64_t max_train_time)
{
	// Parameters for updating the iterates.
	float64_t eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	float64_t sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4.;

	int32_t i, cg_iter;
	float64_t delta, snorm, one=1.0;
	float64_t alpha, f, fnew, prered, actred, gs;

	/* calling external lib */
	int n = (int) fun_obj->get_nr_variable();
	int search = 1, iter = 1, inc = 1;
	double *s = SG_MALLOC(double, n);
	double *r = SG_MALLOC(double, n);
	double *w_new = SG_MALLOC(double, n);
	double *g = SG_MALLOC(double, n);

	for (i=0; i<n; i++)
		w[i] = 0;

	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	delta = tron_dnrm2(n, g, inc);
	float64_t gnorm1 = delta;
	float64_t gnorm = gnorm1;

	if (gnorm <= eps*gnorm1)
		search = 0;

	iter = 1;

	CSignal::clear_cancel();
	CTime start_time;

	while (iter <= max_iter && search && (!CSignal::cancel_computations()))
	{
		if (max_train_time > 0 && start_time.cur_time_diff() > max_train_time)
		  break;

		cg_iter = trcg(delta, g, s, r);

		memcpy(w_new, w, sizeof(float64_t)*n);
		tron_daxpy(n, one, s, inc, w_new, inc);

		gs = tron_ddot(n, g, inc, s, inc);
		prered = -0.5*(gs-tron_ddot(n, s, inc, r, inc));
			fnew = fun_obj->fun(w_new);

		// Compute the actual reduction.
	        actred = f - fnew;

		// On the first iteration, adjust the initial step bound.
		snorm = tron_dnrm2(n, s, inc);
		if (iter == 1)
			delta = CMath::min(delta, snorm);

		// Compute prediction alpha*snorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = CMath::max(sigma1, -0.5*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = CMath::min(CMath::max(alpha, sigma1)*snorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = CMath::max(sigma1*delta, CMath::min(alpha*snorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = CMath::max(sigma1*delta, CMath::min(alpha*snorm, sigma3*delta));
		else
			delta = CMath::max(delta, CMath::min(alpha*snorm, sigma3*delta));

		SG_INFO("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter)

		if (actred > eta0*prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(float64_t)*n);
			f = fnew;
		        fun_obj->grad(w, g);

			gnorm = tron_dnrm2(n, g, inc);
			if (gnorm < eps*gnorm1)
				break;
			SG_SABS_PROGRESS(gnorm, -CMath::log10(gnorm), -CMath::log10(1), -CMath::log10(eps*gnorm1), 6)
		}
		if (f < -1.0e+32)
		{
			SG_WARNING("f < -1.0e+32\n")
			break;
		}
		if (CMath::abs(actred) <= 0 && CMath::abs(prered) <= 0)
		{
			SG_WARNING("actred and prered <= 0\n")
			break;
		}
		if (CMath::abs(actred) <= 1.0e-12*CMath::abs(f) &&
		    CMath::abs(prered) <= 1.0e-12*CMath::abs(f))
		{
			SG_WARNING("actred and prered too small\n")
			break;
		}
	}

	SG_DONE()

	SG_FREE(g);
	SG_FREE(r);
	SG_FREE(w_new);
	SG_FREE(s);
}

int32_t CTron::trcg(float64_t delta, double* g, double* s, double* r)
{
	/* calling external lib */
	int i, cg_iter;
	int n = (int) fun_obj->get_nr_variable();
	int inc = 1;
	double one = 1;
	double *Hd = SG_MALLOC(double, n);
	double *d = SG_MALLOC(double, n);
	double rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = 0.1* tron_dnrm2(n, g, inc);

	cg_iter = 0;
	rTr = tron_ddot(n, r, inc, r, inc);
	while (1)
	{
		if (tron_dnrm2(n, r, inc) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/tron_ddot(n, d, inc, Hd, inc);
		tron_daxpy(n, alpha, d, inc, s, inc);
		if (tron_dnrm2(n, s, inc) > delta)
		{
			SG_INFO("cg reaches trust region boundary\n")
			alpha = -alpha;
			tron_daxpy(n, alpha, d, inc, s, inc);

			double std = tron_ddot(n, s, inc, d, inc);
			double sts = tron_ddot(n, s, inc, s, inc);
			double dtd = tron_ddot(n, d, inc, d, inc);
			double dsq = delta*delta;
			double rad = sqrt(std*std + dtd*(dsq-sts));
			if (std >= 0)
				alpha = (dsq - sts)/(std + rad);
			else
				alpha = (rad - std)/dtd;
			tron_daxpy(n, alpha, d, inc, s, inc);
			alpha = -alpha;
			tron_daxpy(n, alpha, Hd, inc, r, inc);
			break;
		}
		alpha = -alpha;
		tron_daxpy(n, alpha, Hd, inc, r, inc);
		rnewTrnew = tron_ddot(n, r, inc, r, inc);
		beta = rnewTrnew/rTr;
		tron_dscal(n, beta, d, inc);
		tron_daxpy(n, one, r, inc, d, inc);
		rTr = rnewTrnew;
	}

	SG_FREE(d);
	SG_FREE(Hd);

	return(cg_iter);
}

float64_t CTron::norm_inf(int32_t n, float64_t *x)
{
	float64_t dmax = CMath::abs(x[0]);
	for (int32_t i=1; i<n; i++)
		if (CMath::abs(x[i]) >= dmax)
			dmax = CMath::abs(x[i]);
	return(dmax);
}
