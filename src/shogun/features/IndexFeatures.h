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
 *  contain the index of the features. This features used in
 *  the CCustomKernel::init to make the subset of the kernel
 *  matrix.
 *
 * This is used in the CCustomKernel.*/
class CIndexFeatures : public CDummyFeatures
{
	public:
		/** default constructor  */
		CIndexFeatures();

		/** constructor
		 *
		 * @param feature_index feature index vector
		 */
		CIndexFeatures(SGVector<index_t> feature_index);

		/** copy constructor
		 *
		 * @param orig reference to a CIndexFeatures instance
		 */
		CIndexFeatures(const CIndexFeatures &orig);

		/** destructor */
		virtual ~CIndexFeatures();

		/** get number of feature vectors
		 *
		 * @return number of feature vectors
		 */
		virtual int32_t get_num_vectors() const;

		/** duplicate features
		 *
		 * @return the copy of this CIndexFeatures
		 */
		virtual CFeatures* duplicate() const;

		/** get feature type (INT)
		 *
		 * @return F_INT
		 */
		virtual EFeatureType get_feature_type() const;

		/** get feature class (INDEX)
		 *
		 * @return C_INDEX
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get the name of the class
		 *
		 * @return object name
		 */
		virtual const char* get_name() const { return "IndexFeatures"; }

		/** Getter the feature index
		 *
		 * a copy of m_feature_index without subset
		 * a copy of sub_feature_index with subset
		 *
		 * @return matrix feature index
		 */
		SGVector<index_t> get_feature_index();

		/** Setter for feature index
		 *
		 * any subset is removed
		 * set the m_feature_index
		 *
		 * @param feature_index feature index to set
		 *
		 */
		void set_feature_index(SGVector<index_t> feature_index);

		/** free feature index
		 *
		 * Any subset is removed
		 */
		void free_feature_index();

	private:
		/** Initialize CIndexFeatures
		 *
		 * set num_vectors to 0
		 * add m_feature_index to m_parameters
		 */
		void init();

	protected:
		/** feature index */
		SGVector<index_t> m_feature_index;
};
}
#endif
