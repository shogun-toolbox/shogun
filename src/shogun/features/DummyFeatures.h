/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Vladislav Horbatiuk, Evgeniy Andreev, Yuyu Zhang, 
 *          Heiko Strathmann
 */

#ifndef _DUMMYFEATURES__H__
#define _DUMMYFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>

namespace shogun
{
/** @brief The class DummyFeatures implements features that only know the
 * number of feature objects (but don't actually contain any).
 *
 * This is used in the CustomKernel.*/
class DummyFeatures : public Features
{
	public:
		/** default constructor  */
		DummyFeatures();

		/** constructor
		 *
		 * @param num number of feature vectors
		 */
		DummyFeatures(int32_t num);

		/** copy constructor */
		DummyFeatures(const DummyFeatures &orig);

		/** destructor */
		virtual ~DummyFeatures();

		/** get number of feature vectors */
		virtual int32_t get_num_vectors() const;

		/** duplicate features */
		virtual std::shared_ptr<Features> duplicate() const;

		/** get feature type (ANY) */
		inline EFeatureType get_feature_type() const;

		/** get feature class (ANY) */
		virtual EFeatureClass get_feature_class() const;

		/** @return object name */
		virtual const char* get_name() const { return "DummyFeatures"; }

	private:
		void init();

	protected:
		/** number of feature vectors */
		int32_t num_vectors;
};
}
#endif
