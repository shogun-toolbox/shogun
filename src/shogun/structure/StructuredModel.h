/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTURED_MODEL__H__
#define _STRUCTURED_MODEL__H__

#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>

namespace shogun
{

class CStructuredModel;

/** output of the argmax function */
struct CResultSet : public CSGObject
{
	/** destructor */
	~CResultSet() { SG_UNREF(argmax) };

	/** argmax */
	CStructuredData* argmax;

	/** joint feature vector for the given truth */
	SGVector< float64_t > psi_truth;

	/** joint feature vector for the prediction */
	SGVector< float64_t > psi_pred;

	/** corresponding score */
	float64_t score;

	/** delta loss for the prediction vs. truth */
	float64_t delta;

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "ResultSet"; }
};

/** 
 * @brief Class CStructuredModel that represents the application specific model 
 * and contains most of the application dependent logic to solve structured 
 * output (SO) problems. The idea of this class is to be instantiated giving
 * pointers to the functions that are dependent on the application, i.e. the 
 * combined feature representation \f$\Psi(\bold{x},\bold{y})\f$ and the argmax
 * function \f$ {\arg\max} _{\bold{y} \neq \bold{y}_i} \left \langle { \bold{w}, 
 * \Psi(\bold{x}_i,\bold{y}) }  \right \rangle \f$. See: TODO pointer to an 
 * example of these functions is implemented, e.g. for HM-SVM and TODO reference
 * to the paper.
 */
class CStructuredModel : public CSGObject
{
	public:
		/** default constructor */
		CStructuredModel();

		/** constructor
		 *
		 * @param features the feature vectors
		 * @param labels structured labels
		 */
		CStructuredModel(CFeatures* features, CStructuredLabels* labels);

		/** destructor */
		virtual ~CStructuredModel();

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
				SGMatrix < float64_t >  & C);

		/**
		 * return the dimensionality of the joint feature space, i.e. 
		 * the dimension of the weight vector \f$w\f$
		 */
		virtual int32_t get_dim() const = 0;

		/** set labels
		 *
		 * @param labs labels
		 */
		void set_labels(CStructuredLabels* labs);

		/** set features
		 *
		 * @param feats features
		 */
		void set_features(CFeatures* feats);

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

		/** obtains the argmax
		 *
		 * @param w weight vector
		 * @param feat_idx index of the feature to compute the argmax
		 *
		 * @return structure with the predicted output
		 */
		virtual CResultSet* argmax(SGVector< float64_t > w, int32_t feat_idx) = 0;

		/** computes \f$ \Delta(y_{\text{true}}, y_{\text{pred}}) \f$
		 *
		 * @param ytrue_idx index of the true label in labels
		 * @param ypred the predicted label
		 *
		 * @return loss value
		 */
		virtual float64_t delta_loss(int32_t ytrue_idx, CStructuredData* ypred) = 0;

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "StructuredModel"; }
	
		/**
		 * method to be called from a SO machine before training
		 * to ensure that the training data is valid (e.g. check that
		 * there is at least one example for every class). In this class
		 * the method is empty and it can be re-implemented for any
		 * application (e.g. HM-SVM).
		 */
		virtual bool check_training_setup() const;

	private:
		/** internal initialization */
		void init();

	protected:
		/** structured labels */
		CStructuredLabels* m_labels;

		/** feature vectors */
		CFeatures* m_features;

}; /* class CStructuredModel */

} /* namespace shogun */

#endif /* _STRUCTURED_MODEL__H__ */
