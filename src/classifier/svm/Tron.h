#ifndef _CTron_H
#define _CTron_H

#include "base/SGObject.h"

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
};

class CTron : public CSGObject
{
public:
	CTron(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~CTron();

	void tron(double *w);

private:
	int trcg(double delta, double *g, double *s, double *r);
	double norm_inf(int n, double *x);

	double eps;
	int max_iter;
	function *fun_obj;
};

#endif
