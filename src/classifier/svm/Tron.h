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
	virtual float64_t fun(float64_t *w) = 0 ;

	/** grad
	 *
	 * abstract base method
	 *
	 * @param w w
	 * @param g g
	 */
	virtual void grad(float64_t *w, float64_t *g) = 0 ;

	/** Hv
	 *
	 * abstract base method
	 *
	 * @param s s
	 * @param Hs hs
	 */
	virtual void Hv(float64_t *s, float64_t *Hs) = 0 ;

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
	CTron(
		const function *fun_obj, float64_t eps = 0.1, int32_t max_iter = 1000);
	~CTron();

	/** tron
	 *
	 * @param w w
	 */
	void tron(float64_t *w);

	/** @return object name */
	inline virtual const char* get_name() const { return "LibLinear"; }

private:
	int32_t trcg(float64_t delta, double* g, double* s, double* r);
	float64_t norm_inf(int32_t n, float64_t *x);

	float64_t eps;
	int32_t max_iter;
	function *fun_obj;
};

#endif
#endif //HAVE_LAPACK
