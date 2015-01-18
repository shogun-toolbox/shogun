/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#ifndef __FACTORGRAPH_FEATURES_H__
#define __FACTORGRAPH_FEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/structure/FactorGraph.h>

namespace shogun
{

/** @brief CFactorGraphFeatures maintains an array of factor graphs,
 * each graph is a sample, i.e. an instance of structured input.
 */
class CFactorGraphFeatures : public CFeatures
{
	public:
		/** default constructor */
		CFactorGraphFeatures();

		/** constructor
		 *
		 * @param num_samples the number of examples the object will contain
		 */
		CFactorGraphFeatures(int32_t num_samples);

		virtual ~CFactorGraphFeatures();

		/** Copy-constructor
		 *
		 * @return the copy of the given object
		 */
		virtual CFeatures* duplicate() const;

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
		virtual const char* get_name() const { return "FactorGraphFeatures"; }

		/** add a graph instance
		 *
		 * @param fg a factor graph instance
		 * @return whether the sample has been added successfully
		 */
		bool add_sample(CFactorGraph* fg);

		/** get a graph instance
		 *
		 * @param idx index of the required example
		 * @return pointer of CFactorGraph
		 */
		CFactorGraph* get_sample(index_t idx);

		/** helper method used to specialize a base class instance
		 *
		 * @param base_feats its dynamic type must be CFactorGraphFeatures
		 * @return pointer to CFactorGraphFeatures
		 */
		static CFactorGraphFeatures* obtain_from_generic(CFeatures* base_feats);

	protected:
		/** array of CFactorGraph */
		CDynamicObjectArray* m_samples;

	private:
		/** init function for the object */
		void init();
};

}

#endif /* __FACTORGRAPH_FEATURES_H__ */
