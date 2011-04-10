/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef GNB_H_
#define GNB_H_

#include "Classifier.h"

namespace shogun {

/** @brief Class GNB, a Gaussian Naive Bayes classifier
 *
 *	Formally, chooses class c with maximum \f$ P(c)P(x|c) \f$
 *	probability. Naive bayes assumes \f$ P(x|c) \f$ as product
 *  \f$ \prod_i P(x_i|c) \f$ and gaussian naive bayes make assume of
 *  \f$P(x_i|c) \sim \mathcal{N} (\mu_c, \sigma_c)\f$, where \f$ \mathcal{N} \f$ is
 *  normal distribution and \f$ \mu_c, \sigma_c \f$ are estimates of i-th
 *  feature mean and standard deviation.
 *
 */

class CGNB : public CClassifier
{
public:
	/** default constructor
	 *
	 */
	CGNB();

	/** constructor
	 * 	@param train_examples train examples
	 *	@param train_labels labels corresponding to train_examples
	 */
	CGNB(CFeatures* train_examples, CLabels* train_labels);

	/** destructor
	 *
	 */
	virtual ~CGNB();

	/** train classifier
	 * 	@param data train examples
	 * 	@return true if successful
	 */
	virtual bool train(CFeatures* data = NULL);

	/** classify all examples
	 *  @return labels
	 */
	virtual CLabels* classify();

	/** classify specified examples
	 * 	@param data examples to be classified
	 * 	@return labels corresponding to data
	 */
	virtual CLabels* classify(CFeatures* data);

	/** classifiy specified example
	 * 	@param idx example index
	 * 	@return label
	 */
	virtual float64_t classify_example(int32_t idx);

	/**
	 * 	@return classifier name
	 */
	virtual inline const char* get_name() const { return "Gaussian Naive Bayes"; };

	/**
	 * 	@return classifier type
	 */
	virtual inline EClassifierType get_classifier_type() { return CT_GNB; };

protected:

	/// number of train labels
	int32_t num_train_labels;

	/// number of different classes (labels)
	int32_t num_classes;

};

}

#endif /* GNB_H_ */
