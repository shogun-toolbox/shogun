/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Vladislav Horbatiuk, Yuyu Zhang
 */

#ifndef __LATENTFEATURES_H__
#define __LATENTFEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/lib/Data.h>

namespace shogun
{
	/** @brief Latent Features class
	 * The class if for representing features for latent learning, e.g. LatentSVM.
	 * It's basically a very generic way of storing features of any (user-defined) form
	 * based on Data.
	 */
	class LatentFeatures : public Features
	{
		public:
			/** default constructor */
			LatentFeatures();

			/** constructor
			 *
			 * @param num_samples the number of examples the object will contain
			 */
			LatentFeatures(int32_t num_samples);

			virtual ~LatentFeatures();

			/** Copy-constructor
			 *
			 * @return the copy of the given object
			 */
			virtual std::shared_ptr<Features> duplicate() const;

			/** get feature type
			 *
			 * @return templated feature type
			 */
			virtual EFeatureType get_feature_type() const;

			/** get feature class
			 *
			 * @return feature class
			 */
			virtual EFeatureClass get_feature_class() const;

			/** get number of examples
			 *
			 * @return number of examples/vectors (possibly of subset, if implemented)
			 */
			virtual int32_t get_num_vectors() const;

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentFeatures"; }

			/** add latent example
			 *
			 * @param example the user defined Data
			 */
			bool add_sample(const std::shared_ptr<Data>& example);

			/** get latent example
			 *
			 * @param idx index of the required example
			 * @return the user defined Data at the given index
			 */
			std::shared_ptr<Data> get_sample(index_t idx);

			/** helper method used to specialize a base class instance
			 *
			 * @param base_feats its dynamic type must be LatentFeatures
			 */
			static std::shared_ptr<LatentFeatures> obtain_from_generic(const std::shared_ptr<Features>& base_feats);
		protected:
			/** array of Data */
			std::vector<std::shared_ptr<Data>> m_samples;

		private:
			/** init function for the object */
			void init();
	};
}

#endif /* __LATENTFEATURES_H__ */

