/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Evgeniy Andreev, Yuyu Zhang, 
 *          Sergey Lisitsyn, Bjoern Esser, Saurabh Goyal
 */

#ifndef _PNORM_ONE__H__
#define _PNORM_ONE__H__

#include <shogun/lib/config.h>

#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>


namespace shogun
{
/** @brief Preprocessor PNorm, normalizes vectors to have p-norm.
 *
 * Formally, it computes
 *
 * \f[
 * {\bf x} \leftarrow \frac{{\bf x}}{||{\bf x}||_p}
 * \f]
 *
 */
class PNorm : public DensePreprocessor<float64_t>
{
	public:
		/** default PNorm Constructor */
		PNorm ();

    /** constructor
     * @param p the norm to calculate. NOTE: has to be greater or equal than 1.0
     */
		PNorm (double p);

		/** destructor */
		virtual ~PNorm ();

		/// cleanup
		virtual void cleanup ();
		/// initialize preprocessor from file
		virtual bool load (FILE* f);
		/// save preprocessor init-data to file
		virtual bool save (FILE* f);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector (SGVector<float64_t> vector);

		/** @return object name */
		virtual const char* get_name () const { return "PNorm"; }

		/// return a type of preprocessor
		virtual EPreprocessorType get_type () const { return P_PNORM; }

		/**
		 * Set norm
		 * @param pnorm norm value
		 */
		void set_pnorm (double pnorm);

		/**
		 * Get norm value
		 * @return norm
		 */
		double get_pnorm () const;

	protected:
		virtual SGMatrix<float64_t> apply_to_matrix(SGMatrix<float64_t> matrix);

	private:
		void register_param ();
		inline float64_t get_pnorm (float64_t* vec, int32_t vec_len) const;

	private:
		double m_p;
};
}
#endif /* _PNORM_ONE__H__ */
