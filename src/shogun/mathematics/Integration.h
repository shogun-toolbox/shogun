/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (w) 2014 Wu Lin
 * Written (W) 2013 Roman Votyakov
 *
 * The abscissae and weights for Gauss-Kronrod rules are taken form
 * QUADPACK, which is in public domain.
 * http://www.netlib.org/quadpack/
 *
 * See method comments which functions are adapted from GNU Octave,
 * file quadgk.m: Copyright (C) 2008-2012 David Bateman under GPLv3
 * http://www.gnu.org/software/octave/
 *
 * See method comments which functions are adapted from
 * Gaussian Process Machine Learning Toolbox, file util/gauher.m,
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 *
 */

#ifndef _INTEGRATION_H_
#define _INTEGRATION_H_

#include <shogun/lib/config.h>


#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Function.h>

namespace shogun
{
template<class T> class SGVector;

/** @brief Class that contains certain methods related to numerical
 * integration
 */
class CIntegration : public CSGObject
{
public:
	/** numerically evaluate definite integral \f$\int_a^b f(x) dx\f$,
	 * where \f$f(x)\f$ - function of one variable, using adaptive
	 * Gauss-Kronrod quadrature formula
	 *
	 * \f[
	 * \int_a^b f(x)\dx \approx \sum_{i=1}^n w_i f(x_i)
	 * \f]
	 *
	 * where x_i and w_i - Gauss-Kronrod nodes and weights
	 * respectively.
	 *
	 * This function applies the Gauss-Kronrod 21-point integration
	 * rule for finite bounds \f$[a, b]\f$ and 15-point rule for
	 * infinite ones.
	 *
	 * Based on ideas form GNU Octave (file quadgk.m) under GPLv3.
	 *
	 * @param f integrable function of one variable
	 * @param a lower bound of the domain of integration
	 * @param b upper bound of the domain of integration
	 * @param abs_tol absolute tolerance of the quadrature
	 * @param rel_tol relative tolerance of the quadrature
	 * @param max_iter maximum number of iterations of the method
	 * @param sn initial number of subintervals
	 *
	 * @return approximate value of definite integral of the function
	 * on given domain
	 */
	static float64_t integrate_quadgk(CFunction* f, float64_t a,
			float64_t b, float64_t abs_tol=1e-10, float64_t rel_tol=1e-5,
			uint32_t max_iter=1000, index_t sn=10);

	/** numerically evaluate integral of the following kind
	 *
	 * \f[
	 * \int_{-\infty}^{\infty}e^{-x^2}f(x)dx
	 * \f]
	 *
	 * using 64-point Gauss-Hermite rule
	 *
	 * \f[
	 * \int_{-\infty}^{\infty}e^{-x^2}f(x)dx \approx
	 * \sum_{i=1}^{64} w_if(x_i)
	 * \f]
	 *
	 * where x_i and w_i - ith node and weight for the 64-point
	 * Gauss-Hermite formula respectively.
	 *
	 * @param f integrable function of one variable
	 *
	 * @return approximate value of the
	 * integral \f$\int_{-\infty}^{\infty}e^{-x^2}f(x)dx\f$
	 */
	static float64_t integrate_quadgh(CFunction* f);

	/** numerically evaluate integral of the following kind
	 *
	 * \f[
	 * \int_{-\infty}^{\infty}e^{-x^2}f(x)dx
	 * \f]
	 *
	 * using provided Gauss-Hermite points
	 *
	 * \f[
	 * \int_{-\infty}^{\infty}e^{-x^2}f(x)dx \approx
	 * \sum_{i=1}^{64} w_if(x_i)
	 * \f]
	 *
	 * where x_i and w_i - ith node and weight for the provided
	 * Gauss-Hermite formula respectively.
	 *
	 * @param f integrable function of one variable
	 * @param xgh the provided array of nodes
	 * @param wgh the provided array of weights
	 *
	 * @return approximate value of the
	 * integral \f$\int_{-\infty}^{\infty}e^{-x^2}f(x)dx\f$
	 */

	static float64_t integrate_quadgh_customized(CFunction* f,
		SGVector<float64_t> xgh, SGVector<float64_t> wgh);


	/** generate Gauss-Hermite nodes
	 *
	 * Adapted form Gaussian Process Machine Learning Toolbox
	 * (file util/gauher.m)
	 *
	 * @param xgh nodes are saved in this pre-allocated array
	 * @param wgh weights are saved in this pre-allocated array
	 *
	 */
	static void generate_gauher(SGVector<float64_t> xgh, SGVector<float64_t> wgh);


	/** generate 20 Gauss-Hermite nodes using precomputed result
	 *
	 * Adapted form Gaussian Process Machine Learning Toolbox
	 * (file util/gauher.m)
	 *
	 * @param xgh nodes are saved in this pre-allocated array
	 * @param wgh weights are saved in this pre-allocated array
	 *
	 */
	static void generate_gauher20(SGVector<float64_t> xgh, SGVector<float64_t> wgh);

	/** get object name
	 *
	 * @return name Integration
	 */
	virtual const char* get_name() const { return "Integration"; }

private:
	/** evaluate definite integral of a function and error on each
	 * subinterval using Gauss-Kronrod quadrature formula of order n
	 *
	 * Adapted form GNU Octave (file quadgk.m) under GPLv3.
	 *
	 * @param f integrable function of one variable
	 * @param subs subintervals of integration
	 * @param q approximate value of definite integral of the function
	 * on each subinterval
	 * @param err error on each subinterval
	 * @param n order of the Gauss-Kronrod rule
	 * @param xgk Gauss-Kronrod nodes
	 * @param wg Gauss weights
	 * @param wgk Gauss-Kronrod weights
	 */
	static void evaluate_quadgk(CFunction* f, CDynamicArray<float64_t>* subs,
			CDynamicArray<float64_t>* q, CDynamicArray<float64_t>* err, index_t n,
			float64_t* xgk, float64_t* wg, float64_t* wgk);

	/** evaluate definite integral of a function and error on each
	 * subinterval using Gauss-Kronrod quadrature formula of order 15.
	 *
	 * Gauss-Kronrod nodes, Gauss weights and Gauss-Kronrod weights
	 * are precomputed.
	 *
	 * The abscissae and weights for 15-point rule are taken from from
	 * QUADPACK (file dqk15.f).
	 *
	 * @param f integrable function of one variable
	 * @param subs subintervals of integration
	 * @param q approximate value of definite integral of the function
	 * on each subinterval
	 * @param err error on each subinterval
	 */
	static void evaluate_quadgk15(CFunction* f, CDynamicArray<float64_t>* subs,
			CDynamicArray<float64_t>* q, CDynamicArray<float64_t>* err);

	/** evaluate definite integral of a function and error on each
	 * subinterval using Gauss-Kronrod quadrature formula of order 21.
	 *
	 * Gauss-Kronrod nodes, Gauss weights and Gauss-Kronrod weights
	 * are precomputed.
	 *
	 * The abscissae and weights for 21-point rule are taken from
	 * QUADPACK (file dqk21.f).
	 *
	 * @param f integrable function of one variable
	 * @param subs subintervals of integration
	 * @param q approximate value of definite integral of the function
	 * on each subinterval
	 * @param err error on each subinterval
	 */
	static void evaluate_quadgk21(CFunction* f, CDynamicArray<float64_t>* subs,
			CDynamicArray<float64_t>* q, CDynamicArray<float64_t>* err);

	/** evaluate integral \f$\int_{-\infty}^{\infty}e^{-x^2}f(x)dx\f$
	 * using Gauss-Hermite quadrature formula of order n
	 *
	 * @param f integrable function of one variable
	 * @param n order of the Gauss-Hermite rule
	 * @param xh Gauss-Hermite nodes
	 * @param wh Gauss-Hermite weights
	 *
	 * @return approximate value of the integral
	 * \f$\int_{-\infty}^{\infty}e^{-x^2}f(x)dx\f$
	 */
	static float64_t evaluate_quadgh(CFunction* f, index_t n, float64_t* xh,
			float64_t* wh);

	/** evaluate integral \f$\int_{-\infty}^{\infty}e^{-x^2}f(x)dx\f$
	 * using Gauss-Hermite quadrature formula of order 64.
	 *
	 * Gauss-Hermite nodes \f$x_i\f$ and weights \f$w_i\f$ are
	 * precomputed: \f$x_i\f$ - the i-th zero of \f$H_n(x)\f$,
	 * \f$w_i=\frac{2^{n-1}n!\sqrt{\pi}}{n^2[H_{n-1}(x_i)]^2}\f$,
	 * where \f$H_n(x)\f$ is physicists' Hermite polynomials.
	 *
	 * @param f integrable function of one variable
	 *
	 * @return approximate value of the integral
	 * \f$\int_{-\infty}^{\infty}e^{-x^2}f(x)dx\f$
	 */
	static float64_t evaluate_quadgh64(CFunction* f);
};
}
#endif /* _INTEGRATION_H_ */
