#include "lib/config.h"

#ifdef HAVE_LAPACK
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "classifier/svm/SVM_linear.h"
#include "classifier/svm/Tron.h"

l2_lr_fun::l2_lr_fun(const problem *p, float64_t Cp, float64_t Cn)
: function()
{
	int32_t i;
	int32_t l=p->l;
	int32_t *y=p->y;

	this->prob = p;

	z = new float64_t[l];
	D = new float64_t[l];
	C = new float64_t[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2_lr_fun::~l2_lr_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
}


float64_t l2_lr_fun::fun(float64_t *w)
{
	int32_t i;
	float64_t f=0;
	int32_t *y=prob->y;
	int32_t l=prob->l;
	int32_t n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        float64_t yz = y[i]*z[i];
		if (yz >= 0)
		        f += C[i]*log(1 + exp(-yz));
		else
		        f += C[i]*(-yz+log(1 + exp(yz)));
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2_lr_fun::grad(float64_t *w, float64_t *g)
{
	int32_t i;
	int32_t *y=prob->y;
	int32_t l=prob->l;
	int32_t n=prob->n;

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<n;i++)
		g[i] = w[i] + g[i];
}

int32_t l2_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2_lr_fun::Hv(float64_t *s, float64_t *Hs)
{
	int32_t i;
	int32_t l=prob->l;
	int32_t n=prob->n;
	float64_t *wa = new float64_t[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2_lr_fun::Xv(float64_t *v, float64_t *res_Xv)
{
	int32_t l=prob->l;
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	for (int32_t i=0;i<l;i++)
	{
		res_Xv[i]=prob->x->dense_dot(i, v, n);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2_lr_fun::XTv(float64_t *v, float64_t *res_XTv)
{
	int32_t l=prob->l;
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	memset(res_XTv, 0, sizeof(float64_t)*prob->n);

	for (int32_t i=0;i<l;i++)
	{
		prob->x->add_to_dense_vec(v[i], i, res_XTv, n);

		if (prob->use_bias)
			res_XTv[n]+=v[i];
	}
}

l2loss_svm_fun::l2loss_svm_fun(const problem *p, float64_t Cp, float64_t Cn)
: function()
{
	int32_t i;
	int32_t l=p->l;
	int32_t *y=p->y;

	this->prob = p;

	z = new float64_t[l];
	D = new float64_t[l];
	C = new float64_t[l];
	I = new int32_t[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2loss_svm_fun::~l2loss_svm_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
	delete[] I;
}

float64_t l2loss_svm_fun::fun(float64_t *w)
{
	int32_t i;
	float64_t f=0;
	int32_t *y=prob->y;
	int32_t l=prob->l;
	int32_t n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        z[i] = y[i]*z[i];
		float64_t d = z[i]-1;
		if (d < 0)
			f += C[i]*d*d;
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2loss_svm_fun::grad(float64_t *w, float64_t *g)
{
	int32_t i;
	int32_t *y=prob->y;
	int32_t l=prob->l;
	int32_t n=prob->n;

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<n;i++)
		g[i] = w[i] + 2*g[i];
}

int32_t l2loss_svm_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2loss_svm_fun::Hv(float64_t *s, float64_t *Hs)
{
	int32_t i;
	int32_t l=prob->l;
	int32_t n=prob->n;
	float64_t *wa = new float64_t[l];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2loss_svm_fun::Xv(float64_t *v, float64_t *res_Xv)
{
	int32_t l=prob->l;
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	for (int32_t i=0;i<l;i++)
	{
		res_Xv[i]=prob->x->dense_dot(i, v, n);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2loss_svm_fun::subXv(float64_t *v, float64_t *res_Xv)
{
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	for (int32_t i=0;i<sizeI;i++)
	{
		res_Xv[i]=prob->x->dense_dot(I[i], v, n);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2loss_svm_fun::subXTv(float64_t *v, float64_t *XTv)
{
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	memset(XTv, 0, sizeof(float64_t)*prob->n);
	for (int32_t i=0;i<sizeI;i++)
	{
		prob->x->add_to_dense_vec(v[i], I[i], XTv, n);
		
		if (prob->use_bias)
			XTv[n]+=v[i];
	}
}

#endif //HAVE_LAPACK
