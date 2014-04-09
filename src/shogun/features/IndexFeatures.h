/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 pl8787
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
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
		virtual EFeatureType get_feature_type() const;

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
		/** feature index */
		SGVector<index_t> feature_index;
};
}
#endif
