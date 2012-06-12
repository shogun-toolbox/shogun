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

#ifdef HAVE_LAPACK

#include <shogun/regression/Regression.h>
#include <shogun/machine/Machine.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/regression/gp/InferenceMethod.h>

namespace shogun
{
/** @brief Class GaussianProcessRegression implements Gaussian Process Regression.
 * Instead of a distribution over weights, the GP specifies a distribution over functions.
 */

class CGaussianProcessRegression : public CMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** constructor
		 *
		 * @param inf Chosen Inference Method
		 * @param data training data
		 * @param lab labels
		 */
		CGaussianProcessRegression(CInferenceMethod* inf,
					   CDenseFeatures<float64_t>* data, CLabels* lab);

		  /** default constructor */
		CGaussianProcessRegression();

		virtual ~CGaussianProcessRegression();
		
		/** set features
		*
		* @param feat features to set
		*/
		virtual inline void set_features(CDotFeatures* feat)
		{
			SG_UNREF(features);
			SG_REF(feat);
			features=feat;
		}
		
		/** get features
		*
		* @return features
		*/
		virtual CDotFeatures* get_features() { SG_REF(features); return features; }
		
		/** set Inference Method
		*
		* @param inf Inference Method
		*/
		inline void set_method(CInferenceMethod* inf) { m_method = inf; };
		
		/** get Inference Method
		*
		* @return Inference Method
		*/
		inline CInferenceMethod* get_method() { SG_REF(m_method); return m_method; };
			
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
		virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);
		
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
		SGVector<float64_t> getCovarianceVector(CFeatures* data);
		
		/** @return object name */
		inline virtual const char* get_name() const { return "GaussianProcessRegression"; }
	
	protected:
  		/** train regression
		 *
		 * @param data training data 
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);
	private:

		/** function for initialization*/
		void init();

		/** apply mean prediction from data
		*
		* @param data (test)data to be classified
		* @return classified labels
		*/
		virtual CRegressionLabels* mean_prediction(CFeatures* data);

	private:

		/** features */
		CDotFeatures* features;
		
		/** Inference Method */
		CInferenceMethod* m_method;

		
};
}
#endif 
#endif /* _GAUSSIANPROCESSREGRESSION_H__ */
