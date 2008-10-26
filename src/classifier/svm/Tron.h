#ifndef _CTron_H
#define _CTron_H

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "base/SGObject.h"

/** class function */
class function
{
public:
	/** fun
	 *
	 * abstract base method
	 *
	 * @param w w
	 * @return something floaty
	 */
	virtual double fun(double *w) = 0 ;

	/** grad
	 *
	 * abstract base method
	 *
	 * @param w w
	 * @param g g
	 */
	virtual void grad(double *w, double *g) = 0 ;

	/** Hv
	 *
	 * abstract base method
	 *
	 * @param s s
	 * @param Hs hs
	 */
	virtual void Hv(double *s, double *Hs) = 0 ;

	/** get nr variable
	 *
	 * abstract base method
	 *
	 * @return something inty
	 */
	virtual int32_t get_nr_variable(void) = 0 ;

	virtual ~function(void){}
};

/** class Tron */
class CTron : public CSGObject
{
public:
	/** constructor
	 *
	 * @param fun_obj object of class function
	 * @param eps eps
	 * @param max_iter max iter
	 */
	CTron(const function *fun_obj, double eps = 0.1, int32_t max_iter = 1000);
	~CTron();

	/** tron
	 *
	 * @param w w
	 */
	void tron(double *w);

private:
	int32_t trcg(double delta, double *g, double *s, double *r);
	double norm_inf(int32_t n, double *x);

	double eps;
	int32_t max_iter;
	function *fun_obj;
};

#endif
#endif //HAVE_LAPACK
