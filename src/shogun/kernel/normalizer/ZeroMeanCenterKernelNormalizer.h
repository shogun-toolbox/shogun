/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Gorden Jemwa
 * Copyright (C) 2010 University of Stellenbosch
 */

#ifndef _ZEROMEANCENTERKERNELNORMALIZER_H___
#define _ZEROMEANCENTERKERNELNORMALIZER_H___

#include <kernel/normalizer/KernelNormalizer.h>

namespace shogun
{
/** @brief ZeroMeanCenterKernelNormalizer centers the kernel in feature space
 *
 * After centering, each feature must have zero mean.  The centered kernel
 * matrix can be expressed in terms of the non-centered version.
 *
 * Denoting the mapping from input space to feature space by \f$\phi:\mathcal{X}\rightarrow\mathcal{F}\f$, the centered square kernel matrix \f$K_c\f$ (with  dimensionality \f$ M \f$)
 *
 * can be expressed in terms of the original matrix \f$K\f$ as follows:
 *
 * \f{eqnarray*}
 * k({\bf x}_i,{\bf x}_j)_c & = & \left(\phi({\bf x}_i) - \frac{1}{m} \sum_{p=1}^M \phi({\bf x}_p)\right) \cdot \left(\phi({\bf x}_j) - \frac{1}{M} \sum_{q=1}^M \phi({\bf x}_q)\right)  \\
 *          & = & K_{ij} - \frac{1}{M} \sum_{p=1}^M K_{pj} - \frac{1}{M} \sum_{q=1}^M K_{iq} + \frac{1}{M^2} \sum_{p=1}^M \sum_{q=1}^M K_{pq} \\
 *          & = & (K - 1_M K - K 1_M + 1_M K 1_M)_{ij}
 * \f}
 *
 *
 * Additionally, let  \f$ K^{t} \f$  be the \f$ L \times M \f$ test matrix describing the similarity between a \f$ L \f$ test instances with \f$M\f$ training instances
 *
 * (defined by a \f$ M x M \f$ kernel matrix \f$ K\f$), the centered testing set kernel matrix is given by
 * \f[
 * K_{c}^t  =  (K - 1'_M K - K^{t} 1_M + 1'_M K 1_M)
 * \f]
 */
class CZeroMeanCenterKernelNormalizer : public CKernelNormalizer
{
	public:
		/** default constructor
		*/
		CZeroMeanCenterKernelNormalizer()
			: CKernelNormalizer(), ktrain_row_means(NULL), num_ktrain(0),
			ktest_row_means(NULL),	num_ktest(0)
		{
			m_parameters->add_vector(&ktrain_row_means, &num_ktrain,
					"num_ktrain", "Train row means.");
			m_parameters->add_vector(&ktest_row_means, &num_ktest,
					"num_ktest","Test row means.");
		}

		/** default destructor */
		virtual ~CZeroMeanCenterKernelNormalizer()
		{
			SG_FREE(ktrain_row_means);
			SG_FREE(ktest_row_means);
		}

		/** initialization of the normalizer
		 * @param k kernel */
		virtual bool init(CKernel* k)
		{
			ASSERT(k)
			int32_t num_lhs=k->get_num_vec_lhs();
			int32_t num_rhs=k->get_num_vec_rhs();
			ASSERT(num_lhs>0)
			ASSERT(num_rhs>0)

			CFeatures* old_lhs=k->lhs;
			CFeatures* old_rhs=k->rhs;

			/* compute mean for each row of the train matrix*/
			k->lhs=old_lhs;
			k->rhs=old_lhs;

			bool r1=alloc_and_compute_row_means(k, ktrain_row_means, num_lhs,num_lhs);

			/* compute mean for each row of the test matrix */
			k->lhs=old_lhs;
			k->rhs=old_rhs;

			bool r2=alloc_and_compute_row_means(k, ktest_row_means, num_lhs,num_rhs);

			/* compute train kernel matrix mean */
			ktrain_mean=0;
			for (int32_t i=0;i<num_lhs;i++)
				ktrain_mean += (ktrain_row_means[i]/num_lhs);

			k->lhs=old_lhs;
			k->rhs=old_rhs;

			return r1 && r2;
		}

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize(
				float64_t value, int32_t idx_lhs, int32_t idx_rhs)
		{
			value += (-ktrain_row_means[idx_lhs] - ktest_row_means[idx_rhs] + ktrain_mean);
			return value;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs)
		{
			SG_ERROR("normalize_lhs not implemented")
			return 0;
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs)
		{
			SG_ERROR("normalize_rhs not implemented")
			return 0;
		}

		/**
		 * alloc and compute the vector containing the row margins of all rows
		 * for a kernel matrix.
		 */
		bool alloc_and_compute_row_means(CKernel* k, float64_t* &v, int32_t num_lhs, int32_t num_rhs)
		{
			SG_FREE(v);
			v=SG_MALLOC(float64_t, num_rhs);

			for (int32_t i=0; i<num_rhs; i++)
			{
				v[i]=0;
				for (int32_t j=0; j<num_lhs; j++)
					v[i] += ( k->compute(j,i)/num_lhs );
			}
			return (v!=NULL);
		}

		/** @return object name */
		virtual const char* get_name() const { return "ZeroMeanCenterKernelNormalizer"; }

	protected:
		/** train row means */
		float64_t* ktrain_row_means;

		/** num k train */
		int32_t num_ktrain;

		/** test row means */
		float64_t* ktest_row_means;

		/** num k test */
		int32_t num_ktest;

		/** train mean */
		float64_t ktrain_mean;
};
}
#endif
