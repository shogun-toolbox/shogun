/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 */

#ifndef SOBI_H_
#define SOBI_H_

#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/lib/SGNDArray.h>
#include <shogun/features/Features.h>
#include <shogun/converter/ica/ICAConverter.h>

namespace shogun
{

class CFeatures;

/** @brief class SOBI
 * 
 * Implements the Second Order Blind Identification (SOBI) 
 * algorithm for Independent Component Analysis (ICA) and 
 * Blind Source Separation (BSS). This algorithm is also 
 * sometime refered to as Temporal Decorrelation Separation
 * (TDSep).
 * 
 * Belouchrani, A., Abed-Meraim, K., Cardoso, J. F., & Moulines, E. (1997). 
 * A blind source separation technique using second-order statistics. 
 * Signal Processing, IEEE Transactions on, 45(2), 434-444.
 * 
 */
class CSOBI: public CICAConverter
{
	public:
		
		/** constructor */
		CSOBI();

		/** destructor */
		virtual ~CSOBI();

		/** apply to features
		 * @param features
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

		/** getter for time sep cov matrices
		 * @return cov matrices
		 */
		SGNDArray<float64_t> get_covs() const;

		/** @return object name */
		virtual const char* get_name() const { return "SOBI"; };

	protected:

		/** init */
		void init();

	private:
		
		/** tau vector */
		SGVector<float64_t> m_tau;
		
		/** cov matrices */
		SGNDArray<float64_t> m_covs;
};	
}
#endif // HAVE_EIGEN3
#endif // SOBI
