/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Sergey Lisitsyn
 */

#ifndef CONJUGATEINDEX_H_
#define CONJUGATEINDEX_H_
#ifdef HAVE_LAPACK
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/SimpleFeatures.h>

namespace shogun
{

class CLabels;
class CDotFeatures;
class CFeatures;

/** @brief conjugate index classifier.
 * Described in:
 *
 * Fursov V., Kulagina I., Kozin N.
 * Building of classifiers based on conjugation indices
 *
 * Currently supports only multiclass problems.
 * Useless for datasets with # of dimensions less than # of class vectors.
 */
class CConjugateIndex : public CMachine
{

public:
	/** default constructor
	 *
	 */
	CConjugateIndex();

	/** constructor
	 * @param train_features train features
	 * @param train_labels labels corresponding to train_examples
	 */
	CConjugateIndex(CFeatures* train_features, CLabels* train_labels);

	/** destructor
	 *
	 */
	virtual ~CConjugateIndex();

	/** set features
	 * @param features features to be set
	 */
	virtual void set_features(CFeatures* features);

	/** get features
	 * @return current features
	 */
	virtual CSimpleFeatures<float64_t>* get_features();

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
	virtual const char* get_name() const { return "ConjugateIndex"; };

	/** get classifier type
	 * @return classifier type
	 */
	virtual EClassifierType get_classifier_type() { return CT_CONJUGATEINDEX; };

protected:

	/** clean-up class matrices */
	void clean_classes();

	/** computes conjugate index between feature_vector and
	 *  label-th class
	 *
	 * @param feature_vector feature vector
	 * @param label label
	 */
	float64_t conjugate_index(SGVector<float64_t> feature_vector, int32_t label);

protected:

	/** number of classes */
	int32_t m_num_classes;

	/** temporary vector */
	SGVector<float64_t> m_feature_vector;

	/** stores class matrices used to compute conjugate indexes */
	SGMatrix<float64_t>* m_classes;

	/** stores features to be used */
	CSimpleFeatures<float64_t>* m_features;

};

}
#endif /* HAVE_LAPACK */
#endif /* CONJUGATEINDEX_H_ */
