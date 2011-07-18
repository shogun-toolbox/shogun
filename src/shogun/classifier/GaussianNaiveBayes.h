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

#include "machine/Machine.h"
#include "features/DotFeatures.h"

namespace shogun {

class CLabels;
class CDotFeatures;
class CFeatures;

/** @brief Class GaussianNaiveBayes, a Gaussian Naive Bayes classifier
 *
 * This classifier assumes that a posteriori conditional probabilities
 * are gaussian pdfs. For each vector gaussian naive bayes chooses
 * the class C with maximal
 * \[
 * P(c) \prod_{i} P(x_i|c)
 * \]
 *
 */
class CGaussianNaiveBayes : public CMachine
{

public:
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
	virtual inline void set_features(CDotFeatures* features)
	{
		 SG_UNREF(m_features);
		 SG_REF(features);
		 m_features = features;
	}

	/** get features for classify
	 * @return current features
	 */
	virtual inline CDotFeatures* get_features()
	{
		SG_REF(m_features);
		return m_features;
	}

	/** train classifier
	 * @param data train examples
	 * @return true if successful
	 */
	virtual bool train(CFeatures* data = NULL);

	/** classify all examples
	 * @return labels
	 */
	virtual CLabels* apply();

	/** classify specified examples
	 * @param data examples to be classified
	 * @return labels corresponding to data
	 */
	virtual CLabels* apply(CFeatures* data);

	/** classifiy specified example
	 * @param idx example index
	 * @return label
	 */
	virtual float64_t apply(int32_t idx);

	/** get name
	 * @return classifier name
	 */
	virtual inline const char* get_name() const { return "GaussianNaiveBayes"; };

	/** get classifier type
	 * @return classifier type
	 */
	virtual inline EClassifierType get_classifier_type() { return CT_GAUSSIANNAIVEBAYES; };

protected:

	/// features for training or classifying
	CDotFeatures* m_features;

	/// minimal label
	int32_t m_min_label;

	/// actual int labels
	int32_t* m_labels;

	/// number of train labels
	int32_t m_num_train_labels;

	/// number of different classes (labels)
	int32_t m_num_classes;

	/// dimensionality of feature space
	int32_t m_dim;

	/// means for normal distributions of features
	float64_t* m_means;

	/// variances for normal distributions of features
	float64_t* m_variances;

	/// a priori probabilities of labels
	float64_t* m_label_prob;

	/** computes gaussian exponent by x, indexes, m_means and m_variances
	 * @param x feature value
	 * @param l_idx index of label
	 * @param f_idx index of feature
	 * @return exponent value
	 */
	float64_t inline normal_exp(float64_t x, int32_t l_idx, int32_t f_idx)
	{
		return CMath::exp(-CMath::sq(x-m_means[m_dim*l_idx+f_idx])/(2*m_variances[m_dim*l_idx+f_idx]));
	}

	/// label rates
	float64_t* m_rates;
};

}

#endif /* GAUSSIANNAIVEBAYES_H_ */
