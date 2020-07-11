/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Bjoern Esser, Shell Hu
 */

#ifndef __FACTORGRAPH_FEATURES_H__
#define __FACTORGRAPH_FEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/structure/FactorGraph.h>

namespace shogun
{

/** @brief FactorGraphFeatures maintains an array of factor graphs,
 * each graph is a sample, i.e. an instance of structured input.
 */
class FactorGraphFeatures : public Features
{
	public:
		/** default constructor */
		FactorGraphFeatures();

		/** constructor
		 *
		 * @param num_samples the number of examples the object will contain
		 */
		FactorGraphFeatures(int32_t num_samples);

		~FactorGraphFeatures() override;

		/** Copy-constructor
		 *
		 * @return the copy of the given object
		 */
		std::shared_ptr<Features> duplicate() const override;

		/** get feature type
		 *
		 * @return templated feature type
		 */
		EFeatureType get_feature_type() const override;

		/** get feature class
		 *
		 * @return feature class
		 */
		EFeatureClass get_feature_class() const override;

		/** get number of examples
		 *
		 * @return number of examples/vectors (possibly of subset, if implemented)
		 */
		int32_t get_num_vectors() const override;

		/** Returns the name of the SGSerializable instance.
		 *
		 * @return name of the SGSerializable
		 */
		const char* get_name() const override { return "FactorGraphFeatures"; }

		/** add a graph instance
		 *
		 * @param fg a factor graph instance
		 * @return whether the sample has been added successfully
		 */
		bool add_sample(const std::shared_ptr<FactorGraph>& fg);

		/** get a graph instance
		 *
		 * @param idx index of the required example
		 * @return pointer of FactorGraph
		 */
		std::shared_ptr<FactorGraph> get_sample(index_t idx);

		/** helper method used to specialize a base class instance
		 *
		 * @param base_feats its dynamic type must be FactorGraphFeatures
		 * @return pointer to FactorGraphFeatures
		 */
		static std::shared_ptr<FactorGraphFeatures> obtain_from_generic(const std::shared_ptr<Features>& base_feats);

	protected:
		/** array of FactorGraph */
		std::vector<std::shared_ptr<FactorGraph>> m_samples;

	private:
		/** init function for the object */
		void init();
};

}

#endif /* __FACTORGRAPH_FEATURES_H__ */
