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

#ifndef FEATURE_SELECTION_H__
#define FEATURE_SELECTION_H__

#include <shogun/lib/config.h>
#include <shogun/preprocessor/Preprocessor.h>

namespace shogun
{

class CFeatures;
class CLabels;
class CSubsetStack;

/** Enum for feature selection algorithms. See class documentation of
 * CFeatureSelection for their descriptions.
 */
enum EFeatureSelectionAlgorithm
{
	BACKWARD_ELIMINATION,
	FORWARD_SELECTION
};

/** Enum for feature removal policy in feature selection algorithms. See
 * class documentation of CFeatureSelection for their descriptions.
 */
enum EFeatureRemovalPolicy
{
	N_SMALLEST,
	PERCENTILE_SMALLEST,
	N_LARGEST,
	PERCENTILE_LARGEST
};

/** @brief Template class CFeatureSelection, base class for all feature
 * selection preprocessors which select a subset of features (dimensions in the
 * feature matrix) to achieve a specified number of dimensions, #m_target_dim
 * from a given set of features. This class showcases all feature selection
 * algorithms via a generic interface. Supported algorithms are specified by
 * the enum #EFeatureSelectionAlgorithm which can be set via set_algorithm()
 * call. Supported wrapper algorithms are
 *
 * - shogun::BACKWARD_ELIMINATION: apply_backward_elimination() method implements
 * this algorithm. This runs inside a loop till it reaches #m_target_dim. In
 * each iteration, inside another loop that runs for all current dimensions we
 * compute measures and store the scores for each current dimension. Based on
 * those measures a number of features are removed at once.
 *
 * - shogun::FORWARD_SELECTION: apply_forward_selection() method implements this
 * algorithm. Inside a loop it adds selected features to an empty feature set
 * till it reaches #m_target_dim. In each iteration, inside another loop that
 * runs for all current dimensions we compute measures and store the scores for
 * each current dimension. Based on those measures a number of features are
 * removed from the original feature set and added to the new feature set.
 *
 * Since all these algorithm cannot be applied for all the feature selection
 * approaches, the method set_algorithm() is kept abstract which is defined
 * in the subclasses as appropriate.
 *
 * The apply() method acts as a wrapper which decides which above methods to
 * call based on the algorithm specified by #m_algorithm. This method makes
 * a deep copy of the original feature object using CSGObject::clone and then
 * performs feature selection on it. The actual memory requirement depends on
 * how copying a dimension subset is handled in CFeatures::copy_dimension_subset
 * implementation.
 *
 * For computing the measures that are used to rank the features for feature
 * selection task, it relies on an abstract method compute_measures() which
 * is defined in the subclasses.
 *
 * Due to the difference in the measure, the removal policy for features can be
 * different which is specified by the #EFeatureRemovalPolicy enum and can be
 * set by set_policy() call. The supported policies are
 *
 * - shogun::N_SMALLEST: Features corresponding to N smallest measures are removed
 * - shogun::PERCENTILE_SMALLEST: Features corresponding to smallest N% measures are
 *   removed
 * - shogun::N_LARGEST: Features corresponding to N largeest measures are removed
 * - shogun::PERCENTILE_LARGEST: Features corresponding to largest N% measures are
 *   removed
 *
 * Note that not all policies can be adapted for a specific feature seleciton
 * approaches. In general, in classes where feature selection is performed by
 * removing the features which correspond to lowest measure, the policies
 * shogun::N_SMALLEST and shogun::PERCENTILE_SMALLEST are appropriate. When features
 * corresponding to highest measures are removed (e.g. training error in a
 * cross-validation scenario), shogun::N_LARGEST and shogun::PERCENTILE_LARGEST are
 * applicable. Therefore, set_policy() is kept abstract and subclasses define
 * this to allow specific policies to be set.
 *
 * Removal of features in each iteration is handled by an abstract method
 * remove_feats() that removes a set of features at once based on the ranks
 * of the features on the measure. Internally this method also updates a subset
 * stack to keep track of the selected features. After calling apply(),
 * the indices of the original features that are selected can be obtained
 * by calling get_selected_feats() method. Please note that the selected
 * features are always kept in the original order.
 *
 * Some of the methods are for internal purpose and are not exposed to the
 * public API. For example,
 *
 * - method precompute() is provided which is intended to be overridden in the
 * subclasses to perform specific tasks that can be completed beforehand in the
 * feature selection algorithms. For example, see CKernelDependenceMaximization.
 *
 * - method adapt_params() is also overridden in the subclasses which tunes
 * the parameters based on current features that are then used to compute the
 * measure.
 */
template <class ST> class CFeatureSelection : public CPreprocessor
{
public:
	/** Default constructor */
	CFeatureSelection();

	/** Destructor */
	virtual ~CFeatureSelection();

	/** Generic interface for applying the feature selection preprocessor.
	 * Acts as a wrapper which decides which actual method to call based on the
	 * algorithm specified.
	 *
	 * @param features the input features
	 * @return the result feature object after applying the preprocessor
	 */
	virtual CFeatures* apply(CFeatures* features);

	/**
	 * Abstract method that is defined in the subclasses to compute the
	 * measures for the provided features based on which feature selection
	 * is performed.
	 *
	 * @param features the features on which the measure has to be computed
	 * @param idx the index that decides which features should we compute
	 * the measure on
	 * @return the measure based on which features are selected
	 */
	virtual float64_t compute_measures(CFeatures* features, index_t idx)=0;

	/**
	 * Abstract method which is defined in the subclasses to handle the removal
	 * of features based on removal policy (see class documentation). This
	 * also updates the subset internally for selected features.
	 *
	 * @param features the features object from which specific features has
	 * to be removed
	 * @param argsorted the argsorted features based on their measures, entry at
	 * 0 being the index of the feature corresponding to the smallest measure.
	 * @return the feature object after removal of features based on the policy
	 */
	virtual CFeatures* remove_feats(CFeatures* features,
			SGVector<index_t> argsorted)=0;

	/** @return indices of selected features */
	SGVector<index_t> get_selected_feats();

	/** @return the feature class, shogun::C_ANY */
	virtual EFeatureClass get_feature_class();

	/** @return feature type */
	virtual EFeatureType get_feature_type();

	/** @return the preprocessor type */
	virtual EPreprocessorType get_type() const;

	/** @param target_dim the target dimension to achieve */
	void set_target_dim(index_t target_dim);

	/** @return the target dimension */
	index_t get_target_dim() const;

	/**
	 * Abstract method which is overridden in the subclasses to set accepted
	 * feature selection algorithm
	 *
	 * @param algorithm the feature selection algorithm to use
	 */
	virtual void set_algorithm(EFeatureSelectionAlgorithm algorithm)=0;

	/** @return the feature removal algorithm being used */
	EFeatureSelectionAlgorithm get_algorithm() const;

	/**
	 * Abstract method which is overridden in the subclasses to set accepted
	 * feature removal policies based on the measure they use
	 *
	 * @param policy the feature removal policy
	 */
	virtual void set_policy(EFeatureRemovalPolicy policy)=0;

	/** @return the feature removal policy being used */
	EFeatureRemovalPolicy get_policy() const;

	/**
	 * Use this method to set the number or percentile of features to be
	 * removed in each iteration.
	 *
	 * @param num_remove number or percentage of features to be removed in
	 * each iteration
	 */
	void set_num_remove(index_t num_remove);

	/** @return number or percentage of features removed in each iteration */
	index_t get_num_remove() const;

	/**
	 * Setter for labels. This method may be overridden in subclasses if
	 * necessary to set some additional parameters associated with it. For
	 * example, in CDependenceMaximization, we need a feature instance of
	 * the labels which is used in the estimator. So this method is overridden
	 * there to internally convert the labels to a dense feature object in
	 * this set call only
	 *
	 * @param labels the labels
	 */
	virtual void set_labels(CLabels* labels);

	/** @return the labels */
	CLabels* get_labels() const;

	/** performs cleanup */
	virtual void cleanup();

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "FeatureSelection";
	}

protected:
	/**
	 * Applies backward elimination algorithm for performing feature selection.
	 * After performing necessary precomputing (defined by subclasses), it
	 * iteratively eliminates a number of features based on a measure until
	 * target dimension is reached.
	 *
	 * @param features the input features
	 * @return the result feature object after applying the preprocessor
	 */
	virtual CFeatures* apply_backward_elimination(CFeatures* features);

	/**
	 * Performs the tasks which can be computed beforehand before the actual
	 * algorithm begins. This method is overridden in the subclasses. Here
	 * it does nothing.
	 */
	virtual void precompute();

	/**
	 * Tunes the parameters which are required to compute the measure based on
	 * current features. Overridden in the subclasses. Here it does nothing.
	 *
	 * @param features the features based on which parameters are needed to be
	 * tuned for computing measures
	 */
	virtual void adapt_params(CFeatures* features);

	/**
	 * Returns the number of features of the provided feature object. Since the
	 * number of features doesn't make sense for all types of features, this
	 * helper method checks whether obtaining a num_features is possible and
	 * then calls get_num_features() on those features after proper type-cast
	 *
	 * @param features the feature object
	 * @return the number of features
	 */
	index_t get_num_features(CFeatures* features) const;

	/** Target dimension */
	index_t m_target_dim;

	/** Wrapper algorithm for feature selection */
	EFeatureSelectionAlgorithm m_algorithm;

	/** Feature removal policy */
	EFeatureRemovalPolicy m_policy;

	/** Number or percentage of features to be removed. When the policy is set
	 * as shogun::N_SMALLEST or shogun::N_LARGEST, this number decides how many features in
	 * an iteration. For shogun::PERCENTILE_SMALLEST or shogun::PERCENTILE_LARGEST, this
	 * decides the percentage of current number of features to be removed in
	 * each iteration
	 */
	index_t m_num_remove;

	/** The labels for the feature vectors */
	CLabels* m_labels;

	/** The indices of features that are selected */
	CSubsetStack* m_subset;

private:
	/** Register params and initialize with default values */
	void init();

};

}
#endif // FEATURE_SELECTION_H__
