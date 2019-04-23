/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Yuyu Zhang, Saurabh Goyal, 
 *          Sergey Lisitsyn
 */

#ifndef _CLOGPLUSONE__H__
#define _CLOGPLUSONE__H__

#include <shogun/lib/config.h>

#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>


namespace shogun
{
/** @brief Preprocessor LogPlusOne does what the name says, it adds one to a dense
 * real valued vector and takes the logarithm of each component of it.
 *
 * \f[
 * {\bf x}\leftarrow \log({\bf x}+{\bf 1})
 * \f]
 * It therefore does not need any initialization. It is most useful in
 * situations where the inputs are counts: When one compares differences of
 * small counts any difference may matter a lot, while small differences in
 * large counts don't. This is what this log transformation controls for.
 */
class LogPlusOne : public DensePreprocessor<float64_t>
{
	public:
		/** default constructor */
		LogPlusOne();

		/** destructor */
		virtual ~LogPlusOne();

		/// cleanup
		virtual void cleanup();
		/// initialize preprocessor from file
		virtual bool load(FILE* f);
		/// save preprocessor init-data to file
		virtual bool save(FILE* f);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** @return object name */
		virtual const char* get_name() const { return "LogPlusOne"; }

		/// return a type of preprocessor
		virtual EPreprocessorType get_type() const { return P_LOGPLUSONE; }

	protected:
		virtual SGMatrix<float64_t> apply_to_matrix(SGMatrix<float64_t> matrix);
};
}
#endif
