/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 pl8787
 */

#ifndef _INDEXFEATURES__H__
#define _INDEXFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DummyFeatures.h>

#include <shogun/lib/SGVector.h>

namespace shogun
{
/** @brief The class IndexFeatures implements features that
 *  have contain the index of the features
 *
 * This is used in the CCustomKernel.*/
class CIndexFeatures : public CDummyFeatures
{
	public:
		/** default constructor  */
		CIndexFeatures();

		/** constructor
		 *
		 * @param num number of feature vectors
		 */
		CIndexFeatures(SGVector<index_t> vector);

		/** copy constructor */
		CIndexFeatures(const CIndexFeatures &orig);

		/** destructor */
		virtual ~CIndexFeatures();

		/** get number of feature vectors */
		virtual int32_t get_num_vectors() const;

		/** duplicate features */
		virtual CFeatures* duplicate() const;

		/** get feature type (INT) */
		inline EFeatureType get_feature_type() const;

		/** get feature class (INDEX) */
		virtual EFeatureClass get_feature_class() const;

		/** @return object name */
		virtual const char* get_name() const { return "IndexFeatures"; }

		/** Getter the feature index
		 *
		 * in-place without subset
		 * a copy with subset
		 *
		 * @return matrix feature index
		 */
		SGVector<index_t> get_feature_index();

		/** Setter for feature index
		 *
		 * any subset is removed
		 *
		 * vlen is number of feature vectors
		 * see below for definition of feature_index
		 *
		 * @param vector feature index to set
		 *
		 */
		void set_feature_index(SGVector<index_t> vector);

		/** free feature index
		 *
		 * Any subset is removed
		 */
		void free_feature_index();

	private:
		void init();

	protected:
		/** number of feature vectors */
		int32_t num_vectors;
		/** feature index */
		SGVector<index_t> feature_index;
};
}
#endif
