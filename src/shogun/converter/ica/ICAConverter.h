/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Thoralf Klein, Bjoern Esser
 */

#ifndef ICACONVERTER_H_
#define ICACONVERTER_H_

#include <shogun/lib/config.h>
#include <shogun/converter/Converter.h>
#include <shogun/features/Features.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

class CFeatures;

/** @brief class ICAConverter
 * Base class for ICA algorithms
 */
class CICAConverter: public CConverter
{
	public:

		/** constructor */
		CICAConverter();

		/** destructor */
		virtual ~CICAConverter();

		/** apply to features
		 * @param features features to embed
		 */
		virtual CFeatures* apply(CFeatures* features, bool inplace = true) = 0;

		/** setter for mixing matrix, if the mixing matrix is set it will be
		 * used as an initial guess if supported by the algorithm
		 * @param mixing_matrix the initial estimate for the mixing matrix
		 */
		void set_mixing_matrix(SGMatrix<float64_t>mixing_matrix);

		/** getter for mixing_matrix
		 * @return mixing_matrix the final estimated mixing matrix
		 */
		SGMatrix<float64_t> get_mixing_matrix() const;

		/** setter for max_iter, the maximum number of iterations
		 * the ICA algorithm will perform if supported
		 * @param iter the number max number of iterations to perform
		 */
		void set_max_iter(int iter);

		/** getter for max_iter
		 * @return max_iter the number max number of iterations to perform
		 */
		int get_max_iter() const;

		/** setter for tol, the convergence tolerance if supported
		 * @param tol the convergence tolerance
		 */
		void set_tol(float64_t tol);

		/** getter for tol
		 * @return tol the convergence tolerance
		 */
		float64_t get_tol() const;

		/** @return object name */
		virtual const char* get_name() const { return "ICAConverter"; };

	protected:

		/** init */
		void init();

		/** mixing_matrix */
		SGMatrix<float64_t> m_mixing_matrix;

		/** max_iter */
		int max_iter;

		/** tol */
		float64_t tol;
};
}
#endif // ICACONVERTER
