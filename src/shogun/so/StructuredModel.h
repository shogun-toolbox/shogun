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
#include <shogun/features/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>

namespace shogun
{

class CStructuredModel;

/** output of the argmax function */
struct CResultSet : public CSGObject
{
	/** joint feature vector for the given truth */
	SGVector< float64_t > psi_truth;

	/** joint feature vector for the prediction */
	SGVector< float64_t > psi_pred;

	/** corresponding score */
	float64_t score;

	/** delta loss for the prediction vs. truth */
	float64_t delta;
};

/** function type to compute combined features */
typedef SGVector< float64_t > (*FCombinedFeature) (CFeatures* features, CStructuredLabels* labels, int32_t feat_idx, int32_t lab_idx);

/** function type to obtain argmax */
typedef CResultSet* (*FArgmax) (CFeatures* features, CStructuredLabels* labels, SGVector< float64_t> w, int32_t feat_idx);

/** 
 * function type to compute the application specific loss 
 * f$\Delta(y_{\text{true}}, y_{\text{pred}})\f$
 */
typedef float64_t (*FDeltaLoss) (CStructuredLabels* labels, CStructuredData ypred, int32_t ytrue_id);

/** 
 * @brief Class StructuredModel that represents the application specific model 
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

		/** TODO */
		CStructuredModel(FCombinedFeature compute_combined_feature, FArgmax argmax, FDeltaLoss compute_delta_loss);

		/** destructor */
		virtual ~CStructuredModel();

		/** initialize the optimization problem */
		virtual void init();

		/**
		 * return the dimensionality of the joint feature space, i.e. the dimension of the
		 * weight vector \f$w\f$
		 */
		virtual int32_t get_dim();

		/** set labels
		 *
		 * @param labs labels
		 */
		void set_labels(CStructuredLabels* labs);

		void set_features(CFeatures* feats);

		/** TODO */
		SGVector< float64_t > compute_combined_feature(int32_t feat_idx, int32_t lab_idx);

		/** TODO */
		CResultSet* argmax(SGVector< float64_t > w, int32_t feat_idx);

		/** TODO */
		float64_t compute_delta_loss(CStructuredLabels* labels, CStructuredData ypred, int32_t ytrue_id);

		/** @return name of SGSerializable */
		inline virtual const char* get_name() const { return "StructuredModel"; }
	
	protected:
		/** structured labels */
		CStructuredLabels* m_labels;

		/** feature vectors */
		CFeatures* m_features;

		/** combined feature representation */
		FCombinedFeature m_compute_combined_feature;

		/** argmax function */
		FArgmax m_argmax;

		/** \f$\Delta\f$ loss function */
		FDeltaLoss m_compute_delta_loss;

		//TODO add A, a, B, b, lb, ub, C - these are set by init

}; /* class CStructuredModel */

} /* namespace shogun */

#endif /* _STRUCTUREDMODEL_H__ */
