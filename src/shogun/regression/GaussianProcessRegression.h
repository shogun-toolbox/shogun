/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef _GAUSSIANPROCESSREGRESSION_H__
#define _GAUSSIANPROCESSREGRESSION_H__

#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#ifdef HAVE_LAPACK

#include <shogun/regression/Regression.h>
#include <shogun/machine/Machine.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/regression/gp/InferenceMethod.h>

namespace shogun
{

class CInferenceMethod;
class CFeatures;
class CLabels;

/** @brief Class GaussianProcessRegression implements Gaussian Process
 * Regression.vInstead of a distribution over weights, the GP specifies
 * a distribution over functions.
 */

class CGaussianProcessRegression : public CMachine
{

	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** GP return type */
		enum EGPReturnType
		{
			GP_RETURN_MEANS,
			GP_RETURN_COV,
			GP_RETURN_BOTH
		};

		/** constructor
		 *
		 * @param inf Chosen Inference Method
		 * @param data training data
		 * @param lab labels
		 */
		CGaussianProcessRegression(CInferenceMethod* inf,
					   CFeatures* data, CLabels* lab);

		  /** default constructor */
		CGaussianProcessRegression();

		virtual ~CGaussianProcessRegression();
		
		/** set features
		*
		* @param feat features to set
		*/
		virtual inline void set_features(CFeatures* feat)
		{
			SG_UNREF(m_features);
			SG_REF(feat);
			m_features = feat;
			update_kernel_matrices();
		}
		
		/** get features
		*
		* @return features
		*/
		virtual CFeatures* get_features()
		{
			SG_REF(m_features);
			return m_features;
		}
		
		/** set Inference Method
		*
		* @param inf Inference Method
		*/
		inline void set_method(CInferenceMethod* inf)
		{
			SG_UNREF(m_method);
			SG_REF(inf);
			m_method = inf;
		};
		
		/** get Inference Method
		*
		* @return Inference Method
		*/
		inline CInferenceMethod* get_method()
		{
			SG_REF(m_method);
			return m_method;
		};
			
		/** load from file
		*
		* @param srcfile file to load from
		* @return if loading was successful
		*/
		virtual bool load(FILE* srcfile);
		
		/** save to file
		*
		* @param dstfile file to save to
		* @return if saving was successful
		*/
		virtual bool save(FILE* dstfile);

		/** set Kernel
		*
		* @param k Kernel
		*/
		void set_kernel(CKernel* k);
		
		/** get kernel
		*
		* @return kernel
		*/
		CKernel* get_kernel();
		
		/** apply regression to data
		*
		* @param data (test)data to be classified
		* @return classified labels
		*/
		virtual CRegressionLabels* apply_regression(CFeatures* data = NULL);
		
		/** get classifier type
		*
		* @return classifier type GaussianProcessRegression
		*/
		inline virtual EMachineType get_classifier_type()
		{
		  return CT_GAUSSIANPROCESSREGRESSION;
		}
		
		/** get covariance vector
		*
		* @return covariance vector
		*/
		SGVector<float64_t> get_covariance_vector();
		
		/** get predicted mean vector
		 *
		* @return predicted mean vector
		*/
		SGVector<float64_t> get_mean_vector();

		/** @return object name */
		inline virtual const char* get_name() const
		{
			return "GaussianProcessRegression";
		}

		/** set return type
		*
		* @param t return type
		*/
		inline void set_return_type(EGPReturnType t)
		{
			m_return = t;
		};

		/** get return type
		*
		* @return return type
		*/

		inline EGPReturnType get_return_type()
		{
			return m_return;
		};

	
	protected:
  		/** train regression
		 *
		 * @param data training data 
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data = NULL);
	private:

		/** function for initialization*/
		void init();

		/* Update kernel matrices */
		void update_kernel_matrices();

	private:

		/** training features */
		CFeatures* m_features;
		
		/** testing features */
		CFeatures* m_data;

		/*Kernel matrix from testing and training
		 * features
		 */
		SGMatrix<float64_t> m_k_trts;

		/*Kernel matrix from testing
		 * features
		 */
		SGMatrix<float64_t> m_k_tsts;

		/** Inference Method */
		CInferenceMethod* m_method;

		/*What should apply_regression return?*/
		EGPReturnType m_return;
};

}

#endif 
#endif
#endif /* _GAUSSIANPROCESSREGRESSION_H__ */
