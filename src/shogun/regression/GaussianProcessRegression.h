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
#include <shogun/features/SimpleFeatures.h>

namespace shogun
{
/** @brief Class GaussianProcessRegression implements Gaussian Process Regression.
 * Instead of a distribution over weights, the GP specifies a distribution over functions.
 * Here we assume noisy observations y:
 *
 * \f[
 *  y = f(x) + \mathcal{N}(0, \sigma^{2})
 * \f]
 * 
 * 
 * In this simple implementation, the regression predicts 
 * using the mean prediction function = K_{test,train} {(K_{train,train} + sigma*I)}^-1 y
 * Where K is the kernel matrix.
 * 
 */

class CGaussianProcessRegression : public CMachine
{
	public:

	/** constructor
	 *
	 * @param sigma variance of the Gaussian observation noise
	 * @param k Kernel for covariance matrix
	 * @param data training data
	 * @param lab labels
	 */
	CGaussianProcessRegression(float64_t sigma, CKernel* k, 
				   CSimpleFeatures<float64_t>* data, CLabels* lab);

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
		
		/** set sigma
		*
		* @param sigma observation noise
		*/
		inline void set_sigma(float64_t sigma) { m_sigma = sigma; };
		
		/** get sigma
		*
		* @return sigma observation noise
		*/
		inline float64_t get_sigma() { return m_sigma; };
			
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
		
		/** apply regression to all objects
		*
		* @return result labels
		*/
		virtual CLabels* apply();
		
		/** apply regression to data
		*
		* @param data (test)data to be classified
		* @return classified labels
		*/
		virtual CLabels* apply(CFeatures* data);
		
		/** apply regression to one example
		*
		* @param num which example to apply to
		* @return classified value
		*/
		virtual float64_t apply(int32_t num);
		
		/** get classifier type
		*
		* @return classifier type GaussianProcessRegression
		*/
		inline virtual EClassifierType get_classifier_type()
		{
		  return CT_GAUSSIANPROCESSREGRESSION;
		}
		
		/** get covariance matrix
		*
		* @return covariance matrix
		*/
		SGMatrix<float64_t> getCovarianceMatrix(CFeatures* data);
		
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
		virtual CLabels* mean_prediction(CFeatures* data);
				
	private:
				
		/** Observation noise alpha */
		float64_t m_sigma;
		
		/** features */
		CDotFeatures* features;
		
		/** kernel */
		CKernel* kernel;
		
		/** Lower triangle Cholesky decomposition of 
		 *  feature matrix
		 */
		SGMatrix<float64_t> m_L;
		
		/** Alpha used for calculation of mean predictions,
		 * solves the equation (K(train,train)+sigma^2*I) alpha = labels.
		 */
		SGVector< float64_t > m_alpha;

		
};
}
#endif 
#endif /* _GAUSSIANPROCESSREGRESSION_H__ */