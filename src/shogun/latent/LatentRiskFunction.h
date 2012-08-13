/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTRISK_FUNCTION_H__
#define __LATENTRISK_FUNCTION_H__

#include <shogun/structure/RiskFunction.h>

namespace shogun
{
	/** @brief: Calculates the risk function for Latent Structural SVM
	 *
   * \f[
	 *   \sum_{i=1)^n \max_{\hat{y},\hat{h} \in YxH}{\mathbf{w} 
   *   \cdot \Psi(x_i, \hat{y}, \hat{h})+\delta{y_i, \hat{y}, \hat{h}}
	 *  - \sum_{i=1)^n \mathbf{w} \cdot \Psi(x_i, y_i, h^*_i)
   * \f]
	 *
	 * For more details see [1]
	 * [1] C.-N. J. Yu and T. Joachims, 
	 *     "Learning structural SVMs with latent variables"
	 *     presented at the Proceedings of the 26th Annual International Conference on Machine Learning,
	 *     New York, NY, USA, 2009, pp. 1169-1176.
	 * http://www.cs.cornell.edu/~cnyu/papers/icml09_latentssvm.pdf
	 *
	 */
	class CLatentRiskFunction: public CRiskFunction
	{
		public:
			/** default constructor */
			CLatentRiskFunction();

			/** destructor */
			virtual ~CLatentRiskFunction();

			/** computes the value of the risk function and sub-gradient at given point
			 *
			 * @param data
			 * @param R
			 * @param subgrad
			 * @param w
       * @param info Helper info for multiple cutting plane models algorithm
			 */
			virtual void risk(void* data, float64_t* R, float64_t* subgrad, float64_t* W, TMultipleCPinfo* info=0);

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentRiskFunction"; }
	};
}

#endif /* __LATENTSORISK_H__ */

