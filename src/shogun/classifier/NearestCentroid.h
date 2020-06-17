/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Philippe Tillet, Sergey Lisitsyn, Viktor Gal, Fernando Iglesias,
 *          Bjoern Esser, Soeren Sonnenburg, Saurabh Goyal
 */

#ifndef _NEAREST_CENTROID_H__
#define _NEAREST_CENTROID_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{

class DistanceMachine;

/** @brief Class NearestCentroid, an implementation of Nearest Shrunk Centroid classifier
 *
 * To define how close examples are
 * NearestCentroid requires a Distance object to work with (e.g., EuclideanDistance ).
 */

class NearestCentroid : public DistanceMachine{

public:

	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_MULTICLASS);

	/**
	 * Default constructor
	 */
	NearestCentroid();

	/** constructor
	 *
	 * @param distance distance
	 * @param trainlab labels for training
	 */
	NearestCentroid(const std::shared_ptr<Distance>& distance);

	/** Destructor
	 */
	~NearestCentroid() override;

	/** Set shrinking constant
	 *
	 * @param shrinking to be set
	 */
	void set_shrinking(float64_t shrinking) {
		m_shrinking = shrinking ;
	}

	/** Get shrinking constant
	 *
	 * @return value of the shrinking constant
	 */
	float64_t get_shrinking() const{
		return m_shrinking;
	}

	/** Get the centroids
	 *
	 * @return Matrix containing the centroids
	 */
	std::shared_ptr<DenseFeatures<float64_t>> get_centroids() const{
		return m_centroids;
	}

	/** Returns the name of the SGSerializable instance.
	 *
	 * @return name of the SGSerializable
	 */
	const char* get_name() const override { return "NearestCentroid"; }

protected:
	/** train Nearest Centroid classifier
	 *
	 * @param data training data (parameter can be avoided if distance or
	 * kernel-based classifiers are used and distance/kernels are
	 * initialized with train data)
	 *
	 * @return whether training was successful
	 */
	bool train_machine(std::shared_ptr<Features> data=NULL) override;

	/** Stores feature data of underlying model.
	 *
	 * Sets centroids as lhs
	 */

private:
	void init();

protected:
	///	number of classes (i.e. number of values labels can take)
	int32_t m_num_classes;

	///	Shrinking parameter
	float64_t m_shrinking = 0;

	///	The centroids of the trained features
	std::shared_ptr<DenseFeatures<float64_t>> m_centroids;

	///	Tells if the classifier has been trained or not
	bool m_is_trained = false;
};

}

#endif
