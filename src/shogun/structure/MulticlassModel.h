/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _MULTICLASS_MODEL__H__
#define _MULTICLASS_MODEL__H__

#include <shogun/structure/StructuredModel.h>

namespace shogun
{

/** @brief TODO */
class CMulticlassModel : public CStructuredModel
{

	public:
		/** default constructor */
		CMulticlassModel();

		/** constructor
		 *
		 * @param features
		 * @param labels
		 */
		CMulticlassModel(CFeatures* features, CStructuredLabels* labels);

		/** destructor */
		virtual ~CMulticlassModel();

		/**
		 * return the dimensionality of the joint feature space, i.e. 
		 * the dimension of the weight vector \f$w\f$
		 */
		virtual int32_t get_dim() const;

		/**
		 * get joint feature vector
		 *
		 * \f[
		 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y})
		 * \f]
		 *
		 * @param feat_idx index of the feature vector to use
		 * @param y structured label to use
		 *
		 * @return the joint feature vector
		 */
		virtual SGVector< float64_t > get_joint_feature_vector(int32_t feat_idx, CStructuredData* y);

		/** obtains the argmax
		 *
		 * @param w weight vector
		 * @param feat_idx index of the feature to compute the argmax
		 *
		 * @return structure with the predicted output
		 */
		virtual CResultSet* argmax(SGVector< float64_t > w, int32_t feat_idx);

		/** computes \f$ \Delta(y_{\text{true}}, y_{\text{pred}}) \f$
		 *
		 * @param ytrue_idx index of the true label in labels
		 * @param ypred the predicted label
		 *
		 * @return loss value
		 */
		virtual float64_t delta_loss(int32_t ytrue_idx, CStructuredData* ypred);

		/** initialize the optimization problem
		 *
		 * @param A
		 * @param a
		 * @param B
		 * @param b
		 * @param lb
		 * @param ub
		 * @param C
		 */
		virtual void init_opt(
				SGMatrix< float64_t > A,  SGVector< float64_t > a,
				SGMatrix< float64_t > B,  SGVector< float64_t > b,
				SGVector< float64_t > lb, SGVector< float64_t > ub,
				SGMatrix < float64_t > & C);

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "MulticlassModel"; }

	private:
		/** internal initilization */
		void init();

}; /* MulticlassModel */

} /* namespace shogun */

#endif /* _MULTICLASS_MODEL__H__ */
