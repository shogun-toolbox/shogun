/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn 
 */

#ifndef DIRECTOR_STRUCTURED_MODEL_H_
#define DIRECTOR_STRUCTURED_MODEL_H_

#include <shogun/structure/StructuredModel.h>
#include <shogun/lib/config.h>
#ifdef USE_SWIG_DIRECTORS
namespace shogun
{

class CStructuredModel;

#define IGNORE_IN_CLASSLIST
/**
 * @brief Class CDirectorStructuredModel that represents the application specific model
 * with structured output implemented in target interface language. It is a base class 
 * that needs to be extended with real implementations before using.
 *
 * @see CStructuredModel
 */
IGNORE_IN_CLASSLIST class CDirectorStructuredModel : public CStructuredModel
{
	public:
		/** default constructor */
		CDirectorStructuredModel();

		/** destructor */
		virtual ~CDirectorStructuredModel();

		/**
		 * return the dimensionality of the joint feature space, i.e.
		 * the dimension of the weight vector \f$w\f$
		 */
		virtual int32_t get_dim() const;

		/**
		 * gets joint feature vector
		 *
		 * \f[
		 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y}_\text{lab\_idx})
		 * \f]
		 *
		 * @param feat_idx index of the feature vector to use
		 * @param lab_idx index of the structured label to use
		 *
		 * @return the joint feature vector
		 */
		SGVector< float64_t > get_joint_feature_vector(int32_t feat_idx, int32_t lab_idx);

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

		/**
		 * obtains the argmax of \f$ \Delta(y_{pred}, y_{truth}) +
		 * \langle w, \Psi(x_{truth}, y_{pred}) \rangle \f$
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

		/** computes \f$ \Delta(y_{\text{true}}, y_{\text{pred}}) \f$
		 *
		 * @param ytrue_idx index of the true label in labels
		 * @param ypred the predicted label
		 *
		 * @return loss value
		 */
		float64_t delta_loss(int32_t ytrue_idx, CStructuredData* ypred);

		/** computes \f$ \Delta(y_{1}, y_{2}) \f$
		 *
		 * @param y1 an instance of structured data
		 * @param y2 another instance of structured data
		 *
		 * @return loss value
		 */
		virtual float64_t delta_loss(CStructuredData* y1, CStructuredData* y2);

		using CStructuredModel::risk;

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "DirectorStructuredModel"; }

}; /* class CDirectorStructuredModel */
} /* namespace shogun */
#endif /* USE_SWIG_DIRECTORS */
#endif /* DIRECTOR_STRUCTURED_MODEL_H_ */
