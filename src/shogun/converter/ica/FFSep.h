/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 */

#ifndef FFSEP_H_
#define FFSEP_H_

#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <lib/SGNDArray.h>
#include <features/Features.h>
#include <converter/ica/ICAConverter.h>

namespace shogun
{

class CFeatures;

/** @brief class FFSep
 *
 * Implements the FFSep algorithm for Independent
 * Component Analysis (ICA) and Blind Source
 * Separation (BSS).
 *
 * Ziehe, A., Laskov, P., Nolte, G., & Müller, K. R. (2004).
 * A fast algorithm for joint diagonalization with non-orthogonal transformations
 * and its application to blind source separation.
 * The Journal of Machine Learning Research, 5, 777-800.
 *
 */
class CFFSep: public CICAConverter
{
	public:

		/** constructor */
		CFFSep();

		/** destructor */
		virtual ~CFFSep();

		/** apply to features
		 * @param features features to embed
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
		virtual const char* get_name() const { return "FFSep"; };

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
#endif // FFSEP
