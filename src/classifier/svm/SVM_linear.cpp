#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "classifier/svm/SVM_linear.h"
#include "classifier/svm/Tron.h"

l2loss_svm_fun::l2loss_svm_fun(const problem *p, double Cp, double Cn)
{
	int i;
	int l=p->l;
	int *y=p->y;

	this->prob = p;

	z = new double[l];
	D = new double[l];
	C = new double[l];
	I = new int[l];

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

double l2loss_svm_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        z[i] = y[i]*z[i];
		double d = z[i]-1;
		if (d < 0)
			f += C[i]*d*d;
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2loss_svm_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

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

int l2loss_svm_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2loss_svm_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	double *wa = new double[l];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2loss_svm_fun::Xv(double *v, double *res_Xv)
{
	int l=prob->l;
	int n=prob->n;

	if (prob->use_bias)
		n--;

	for(int i=0;i<l;i++)
	{
		res_Xv[i]=prob->x->dense_dot(1.0, i, v, n, 0);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2loss_svm_fun::subXv(double *v, double *res_Xv)
{
	int n=prob->n;

	if (prob->use_bias)
		n--;

	for(int i=0;i<sizeI;i++)
	{
		res_Xv[i]=prob->x->dense_dot(1.0, I[i], v, n, 0);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2loss_svm_fun::subXTv(double *v, double *XTv)
{
	int n=prob->n;

	if (prob->use_bias)
		n--;

	memset(XTv, 0, sizeof(double)*prob->n);
	for(int i=0;i<sizeI;i++)
	{
		prob->x->add_to_dense_vec(v[i], I[i], XTv, n);
		
		if (prob->use_bias)
			XTv[n]+=v[i];
	}
}

l2_lr_fun::l2_lr_fun(const problem *p, double Cp, double Cn)
{
	int i;
	int l=p->l;
	int *y=p->y;

	this->prob = p;

	z = new double[l];
	D = new double[l];
	C = new double[l];

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


double l2_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        double yz = y[i]*z[i];
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

void l2_lr_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

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

int l2_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	double *wa = new double[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2_lr_fun::Xv(double *v, double *res_Xv)
{
	int l=prob->l;
	int n=prob->n;

	if (prob->use_bias)
		n--;

	for (int i=0;i<l;i++)
	{
		res_Xv[i]=prob->x->dense_dot(1.0, i, v, n, 0);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2_lr_fun::XTv(double *v, double *res_XTv)
{
	int l=prob->l;
	int n=prob->n;

	if (prob->use_bias)
		n--;

	memset(res_XTv, 0, sizeof(double)*prob->n);

	for (int i=0;i<l;i++)
	{
		prob->x->add_to_dense_vec(v[i], i, res_XTv, n);

		if (prob->use_bias)
			res_XTv[n]+=v[i];
	}
}
