/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Yuyu Zhang, Viktor Gal, Fernando Iglesias, 
 *          Bjoern Esser, Soeren Sonnenburg
 */

#ifndef GAUSSIANNAIVEBAYES_H_
#define GAUSSIANNAIVEBAYES_H_

#include <shogun/lib/config.h>

#include <shogun/machine/NativeMulticlassMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DotFeatures.h>

namespace shogun {

class Labels;
class DotFeatures;
class Features;

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
class GaussianNaiveBayes : public NativeMulticlassMachine
{

public:
	MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

	/** default constructor
	 *
	 */
	GaussianNaiveBayes();

	/** constructor
	 * @param train_examples train examples
	 * @param train_labels labels corresponding to train_examples
	 */
	GaussianNaiveBayes(const std::shared_ptr<Features>& train_examples);

	/** destructor
	 *
	 */
	~GaussianNaiveBayes() override;

	/** set features for classify
	 * @param features features to be set
	 */
	virtual void set_features(std::shared_ptr<Features> features);

	/** get features for classify
	 * @return current features
	 */
	virtual std::shared_ptr<Features> get_features();

	/** classify specified examples
	 * @param data examples to be classified
	 * @return labels corresponding to data
	 */
	std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data=NULL) override;

	/** classifiy specified example
	 * @param idx example index
	 * @return label
	 */
	float64_t apply_one(int32_t idx) override;

	/** get name
	 * @return classifier name
	 */
	const char* get_name() const override { return "GaussianNaiveBayes"; };

	/** get classifier type
	 * @return classifier type
	 */
	EMachineType get_classifier_type() override { return CT_GAUSSIANNAIVEBAYES; };

protected:

	/** train classifier
	 * @param data train examples
	 * @return true if successful
	 */
	bool train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs) override;

private:
	void init();

protected:

	/// features for training or classifying
	std::shared_ptr<DotFeatures> m_features;

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
