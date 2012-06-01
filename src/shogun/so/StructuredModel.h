/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTUREDMODEL_H__
#define _STRUCTUREDMODEL_H__

#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/so/ArgMaxFunction.h>
#include <shogun/so/StructuredLossFunction.h>

namespace shogun
{

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

		/** TODO constructor with members */

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
				SGMatrix < float64_t > C);

		/**
		 * return the dimensionality of the joint feature space, i.e. 
		 * the dimension of the weight vector \f$w\f$
		 */
		virtual int32_t get_dim();

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

		/** computes \f$ \Psi(\bf{x}, \bf{y}) \f$ */
		SGVector< float64_t > compute_joint_feature(int32_t feat_idx, int32_t lab_idx);

		/** obtains the argmax
		 *
		 * @param w weight vector
		 * @param feat_idx index of the feature to compute the argmax
		 *
		 * @return structure with the predicted output
		 */
		CResultSet* argmax(SGVector< float64_t > w, int32_t feat_idx);

		/** computes \f$ \Delta(y_{\text{true}}, y_{\text{pred}}) \f$
		 *
		 * @param labels true labels
		 * @param ytrue_idx index of the true label in labels
		 * @param ypred the predicted label
		 *
		 * @return loss value
		 */
		float64_t compute_delta_loss(CStructuredLabels* labels, int32_t ytrue_idx, CStructuredData ypred);

		/** @return name of SGSerializable */
		inline virtual const char* get_name() const { return "StructuredModel"; }
	
	private:
		/** internal initialization */
		void init();

	protected:
		/** structured labels */
		CStructuredLabels* m_labels;

		/** feature vectors */
		CFeatures* m_features;

		/** argmax function */
		CArgMaxFunction* m_argmax;

		/** \f$\Delta\f$ loss function */
		CStructuredLossFunction* m_loss;

}; /* class CStructuredModel */

} /* namespace shogun */

#endif /* _STRUCTUREDMODEL_H__ */
