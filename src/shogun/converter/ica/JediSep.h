/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 */

#ifndef JEDISEP_H_
#define JEDISEP_H_

#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/converter/Converter.h>
#include <shogun/features/Features.h>

namespace shogun
{

class CFeatures;

/** @brief class JediSep
 * 
 * Implements the JediSep algorithm for Independent 
 * Component Analysis (ICA) and Blind Source 
 * Separation (BSS).
 * 
 * Souloumiac, A. (2009). 
 * Nonorthogonal joint diagonalization by combining givens and hyperbolic rotations. 
 * Signal Processing, IEEE Transactions on, 57(6), 2222-2231.
 *
 */
class CJediSep: public CConverter
{
	public:
		
		/** constructor */
		CJediSep();

		/** destructor */
		virtual ~CJediSep();

		/** apply to features
		 * @param features to embed
		 * @param embedding features
		 */
		virtual CFeatures* apply(CFeatures* features);

		/** getter for tau parameter
		 * @return tau vector
		 */
		SGVector<float64_t> get_tau() const;

		/** setter for tau parameter
		 * @param tau vector
		 */
		void set_tau(SGVector<float64_t> tau);

		/** getter for mixing_matrix
		 * @return mixing_matrix
		 */
		SGMatrix<float64_t> get_mixing_matrix() const;

		/** get name */
		virtual const char* get_name() const { return "FFSep"; };

	protected:

		/** init */
		void init();

	private:
		
		/** tau vector */
		SGVector<float64_t> m_tau;

		/** mixing_matrix */
		SGMatrix<float64_t> m_mixing_matrix;
};	
}
#endif // HAVE_EIGEN3
#endif // JEDISEP
