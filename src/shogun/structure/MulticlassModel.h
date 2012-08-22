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

/**
 * @brief Class CMulticlassModel that represents the application specific model
 * and contains the application dependent logic to solve multiclass
 * classification within a generic SO framework.
 */
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
		 * @param training true if argmax is called during training.
		 * Then, it is assumed that the label indexed by feat_idx in
		 * m_labels corresponds to the true label of the corresponding
		 * feature vector.
		 *
		 * @return structure with the predicted output
		 */
		virtual CResultSet* argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training = true);

		/** computes \f$ \Delta(y_{1}, y_{2}) \f$
		 *
		 * @param y1 an instance of structured data
		 * @param y2 another instance of structured data
		 *
		 * @return loss value
		 */
		virtual float64_t delta_loss(CStructuredData* y1, CStructuredData* y2);

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
				SGMatrix< float64_t > & A,  SGVector< float64_t > a,
				SGMatrix< float64_t > B,  SGVector< float64_t > & b,
				SGVector< float64_t > lb, SGVector< float64_t > ub,
				SGMatrix < float64_t > & C);

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "MulticlassModel"; }

		/** implements risk function for multiclass SO-SVM
		 *
		 * Implementation of risk function for structured output multiclass SVM.
		 *
		 * The value of the risk is evaluated as
		 *
		 * \f[
		 * R({\bf w}) = \sum_{i=1}^{m} \max_{y \in \mathcal{Y}} \left[ \ell(y_i, y)
		 * + \langle {\bf w}, \Psi(x_i, y) - \Psi(x_i, y_i)  \rangle  \right]
		 * \f]
		 *
		 * The subgradient is by Danskin's theorem given as
		 *
		 * \f[
		 * R'({\bf w}) = \sum_{i=1}^{m} \Psi(x_i, \hat{y}_i) - \Psi(x_i, y_i),
		 * \f]
		 *
		 * where \f$ \hat{y}_i \f$ is the most violated label, i.e.
		 *
		 * \f[
		 * \hat{y}_i = \arg\max_{y \in \mathcal{Y}} \left[ \ell(y_i, y)
		 * + \langle {\bf w}, \Psi(x_i, y)  \rangle \right]
		 * \f]
		 *
		 * @param subgrad Subgradient computed at given point W
		 * @param W Given weight vector
		 * @param info Helper info for multiple cutting plane models algorithm
		 * @return Value of the computed risk at given point W
		 */
		virtual float64_t risk(float64_t* subgrad, float64_t* W, TMultipleCPinfo* info=0);

	private:
		void init();

		/** Different flavours of the delta_loss that become handy */
		float64_t delta_loss(float64_t y1, float64_t y2);
		float64_t delta_loss(int32_t y1_idx, float64_t y2);

	private:
		/** number of classes */
		int32_t m_num_classes;

}; /* MulticlassModel */

} /* namespace shogun */

#endif /* _MULTICLASS_MODEL__H__ */
