/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef DEPENDENCE_MAXIMIZATION_H__
#define DEPENDENCE_MAXIMIZATION_H__

#include <shogun/lib/config.h>
#include <shogun/preprocessor/FeatureSelection.h>

namespace shogun
{

class CFeatures;
class CIndependenceTest;

/** @brief Class CDependenceMaximization, base class for all feature selection
 * preprocessors which select a subset of features that shows maximum dependence
 * between the features and the labels. This is done via an implementation of
 * CIndependenceTest, #m_estimator inside compute_measures() (see class
 * documentation of CFeatureSelection), which performs a statistical test for a
 * given feature \f$\mathbf{X}_i\f$ from the set of features \f$\mathbf{X}\f$,
 * and the labels \f$\mathbf{Y}\f$. The test checks
 * \f[
 * 	\textbf{H}_0 : P\left(\mathbf{X}\setminus \mathbf{X}_i, \mathbf{Y}\right)
 *	=P\left(\mathbf{X}\setminus \mathbf{X}_i\right)P\left(\mathbf{Y}\right)
 * \f]
 * The test statistic is then used as a measure which signifies the independence
 * between the rest of the features and the labels - higher the value of the
 * test statistic, greater the dependency between the rest of the features
 * and the class labels, and therefore lesser significant the current feature
 * becomes. Therefore, highest scoring features are removed. The removal policy
 * thus can only be shogun::N_LARGEST and shogun::PERCENTILE_LARGEST and
 * it can be set via set_policy() call. remove_feats() method handles the
 * removal of features based on the specified policy.
 *
 * The estimator cannot be set via user interface, rather its subclasses
 * initialize this estimator with appropriate instances internally.
 *
 * This class also overrides set_labels() method to create a feature object from
 * the labels and sets this as features \f$\mathbf{Y}\sim q\f$ to the estimator
 * which is required to compute the measure.
 */
class CDependenceMaximization : public CFeatureSelection<float64_t>
{
public:
	/** Default constructor */
	CDependenceMaximization();

	/** Destructor */
	virtual ~CDependenceMaximization();

	/**
	 * Method that computes the measures using test statistic computed by
	 * an instance of CIndependenceTest wiht the provided features and
	 * the labels
	 *
	 * @param features the features on which the measure has to be computed
	 * @param idx the index that decides which features should we compute
	 * the measure on
	 * @return the measure based on which features are selected
	 */
	virtual float64_t compute_measures(CFeatures* features, index_t idx);

	/**
	 * Method which handles the removal of features based on removal policy.
	 * see documentation of CFeatureSelection. For dependence maximization
	 * approach, the highest scoring features are removed. Therefore, only
	 * #m_policy can only be shogun::N_LARGEST, shogun::PERCENTILE_LARGEST.
	 * See set_policy() method for specifying the exact policy
	 *
	 * @param features the features object from which specific features has
	 * to be removed
	 * @param ranks the ranks of the features based on their measures, 0 being
	 * the lowest rank which corresponds to smallest measure.
	 * @return the feature object after removal of features based on the policy
	 */
	virtual CFeatures* remove_feats(CFeatures* features, SGVector<index_t> ranks);

	/** @param policy feature removal policy */
	virtual void set_policy(EFeatureRemovalPolicy policy);

	/**
	 * Abstract method which is overridden in the subclasses to set accepted
	 * feature selection algorithm
	 *
	 * @param algorithm the feature selection algorithm to use
	 */
	virtual void set_algorithm(EFeatureSelectionAlgorithm algorithm)=0;

	/**
	 * Initialize preprocessor from features
	 *
	 * @param features the features
	 * @return true if init was successful
	 */
	virtual bool init(CFeatures* features);

	/**
	 * Setter for labels. This method is overridden to internally convert the
	 * labels to a dense feature object and set this feature in the
	 * independence test estimator. These labels serve as samples
	 * \f$\mathbf{Y}\sim q\f$ in the independence test
	 *
	 * @param labels the labels
	 */
	virtual void set_labels(CLabels* labels);

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "DependenceMaximization";
	}

protected:
	/**
	 * Helper method which removes the dimension specified via the index. It
	 * copies the rest of the features into a separate object via
	 * copy_dimension_subset() call.
	 *
	 * @param features the features
	 * @param idx index of the dimension which is required to be removed
	 * @return a new feature object with the specified dimension removed
	 */
	virtual CFeatures* create_transformed_copy(CFeatures* features, index_t idx);

	/**
	 * The estimator for performing statistical tests for independence which
	 * is used for computing measures
	 */
	CIndependenceTest* m_estimator;

	/** The feature for the labels */
	CFeatures* m_labels_feats;

private:
	/** Register params and initialize with default values */
	void init();

};

}
#endif // DEPENDENCE_MAXIMIZATION_H__
