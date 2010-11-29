/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DUMMYFEATURES__H__
#define _DUMMYFEATURES__H__

#include "lib/common.h"
#include "features/Features.h"
#include "base/Parameter.h"

namespace shogun
{
/** @brief The class DummyFeatures implements features that only know the
 * number of feature objects (but don't actually contain any).
 *
 * This is used in the CCustomKernel.*/
class CDummyFeatures : public CFeatures
{
	void init(void) {
		m_parameters->add(&num_vectors, "num_vectors",
						  "Number of feature vectors.");
	}

	public:
		/** default constructor  */
		CDummyFeatures(void) {
			init();

			num_vectors = 0;
		}

		/** constructor
		 *
		 * @param num number of feature vectors
		 */
		CDummyFeatures(int32_t num) : CFeatures(0), num_vectors(num)
		{
			init();
		}

		/** copy constructor */
		CDummyFeatures(const CDummyFeatures &orig) : CFeatures(0),
			num_vectors(orig.num_vectors)
		{
			init();
		}

		/** destructor */
		virtual ~CDummyFeatures()
		{
		}

		/** get number of feature vectors */
		virtual int32_t get_num_vectors()
		{
			return num_vectors;
		}

		/** get size of features (always 1) */
		virtual int32_t get_size()
		{
			return 1;
		}

		/** duplicate features */
		virtual CFeatures* duplicate() const
		{
			return new CDummyFeatures(*this);
		}

		/** get feature type (ANY) */
		inline EFeatureType get_feature_type()
		{
			return F_ANY;
		}

		/** get feature class (ANY) */
		inline virtual EFeatureClass get_feature_class()
		{
			return C_ANY;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "DummyFeatures"; }

	protected:
		/** number of feature vectors */
		int32_t num_vectors;
};
}
#endif
