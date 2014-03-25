/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef GAUSSIANNAIVEBAYES_H_
#define GAUSSIANNAIVEBAYES_H_

#include <shogun/lib/config.h>
#include <shogun/machine/NativeMulticlassMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DotFeatures.h>

namespace shogun {

class CLabels;
class CDotFeatures;
class CFeatures;

/** @brief Class GaussianNaiveBayes, a Gaussian Naive Bayes classifier
 *
 * This classifier assumes that a posteriori conditional probabilities
 * are gaussian pdfs. For each vector gaussian naive bayes chooses
 * the class C with maximal
 *
 * \f[
 * P(c) \prod_{i} P(x_i|c)
 * \f]
 *
 */
class CGaussianNaiveBayes : public CNativeMulticlassMachine
{

public:
	MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

	/** default constructor
	 *
	 */
	CGaussianNaiveBayes();

	/** constructor
	 * @param train_examples train examples
	 * @param train_labels labels corresponding to train_examples
	 */
	CGaussianNaiveBayes(CFeatures* train_examples, CLabels* train_labels);

	/** destructor
	 *
	 */
	virtual ~CGaussianNaiveBayes();

	/** set features for classify
	 * @param features features to be set
	 */
	virtual void set_features(CFeatures* features);

	/** get features for classify
	 * @return current features
	 */
	virtual CFeatures* get_features();

	/** classify specified examples
	 * @param data examples to be classified
	 * @return labels corresponding to data
	 */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

	/** classifiy specified example
	 * @param idx example index
	 * @return label
	 */
	virtual float64_t apply_one(int32_t idx);

	/** get name
	 * @return classifier name
	 */
	virtual const char* get_name() const { return "GaussianNaiveBayes"; };

	/** get classifier type
	 * @return classifier type
	 */
	virtual EMachineType get_classifier_type() { return CT_GAUSSIANNAIVEBAYES; };

protected:

	/** train classifier
	 * @param data train examples
	 * @return true if successful
	 */
	virtual bool train_machine(CFeatures* data=NULL);

protected:

	/// features for training or classifying
	CDotFeatures* m_features;

	/// minimal label
	int32_t m_min_label;

	/// number of different classes (labels)
	int32_t m_num_classes;

	/// dimensionality of feature space
	int32_t m_dim;

	/// means for normal distributions of features
	SGMatrix<float64_t> m_means;

	/// variances for normal distributions of features
	SGMatrix<float64_t> m_variances;

	/// a priori probabilities of labels
	SGVector<float64_t> m_label_prob;

	/// label rates
	SGVector<float64_t> m_rates;
};

}

#endif /* GAUSSIANNAIVEBAYES_H_ */
