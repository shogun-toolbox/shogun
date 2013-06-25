/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2013 Fernando J. Iglesias Garcia
 */

#ifndef LMNN_H_
#define LMNN_H_

#ifdef HAVE_EIGEN3

#include <shogun/base/SGObject.h>
#include <shogun/distance/CustomMahalanobisDistance.h>
#include <shogun/features/Features.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/**
 * @brief Class LMNN that implements the distance metric learning technique
 * Large Margin Nearest Neighbour (LMNN) described in
 *
 * Weinberger, K. Q., Saul, L. K.
 * Distance Metric Learning for Large Margin Nearest Neighbor Classification.
 */
class CLMNN : public CSGObject
{
	public:
		/** default constructor */
		CLMNN();

		/** standard constructor
		 *
		 * @param features feature vectors
		 * @param labels labels of the features
		 */
		CLMNN(CFeatures* features, CMulticlassLabels* labels);

		/** destructor */
		virtual ~CLMNN();

		/** @return name of SGSerializable */
		virtual const char* get_name() const;

		/**
		 * LMNN algorithm to learn a linear transformation of the original feature
		 * space (or, equivalently, a Mahalanobis distance) such that kNN
		 * classification performance is maximized
		 *
		 * @param init_transform initial linear transform
		 */
		void train(SGMatrix<float64_t> init_transform);

		/** get the learnt linear transform (denoted L in LMNN literature typically)
		 *
		 * @return the linear transform L
		 */
		SGMatrix<float64_t> get_linear_transform() const;

		/**
		 * get the learnt Mahalanobis distance (typically denoted M in LMNN literature)
		 * encapsulated in a CCustomMahalanobisDistance object, suitable to be used in kNN
		 *
		 * @return the distance M
		 */
		CCustomMahalanobisDistance* get_distance() const;

	private:
		/** register parameters */
		void init();

	private:
		/** the linear transform learnt by LMNN once train has been called */
		SGMatrix<float64_t> m_linear_transform;

		/** training features */
		CFeatures* m_features;

		/** training labels */
		CLabels* m_labels;

}; /* class CLMNN */

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */

#endif /* LMNN_H_ */
