/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Fernando Iglesias,
 *          Christopher Goldsworthy, Heiko Strathmann, Yuyu Zhang,
 *          Chiyuan Zhang
 */

#ifndef _DENSE_LABELS__H__
#define _DENSE_LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/labels/Labels.h>
#include <shogun/features/SubsetStack.h>

namespace shogun
{
	class CFile;
	class CBinaryLabels;

	/** @brief Dense integer or floating point labels
	 *
	 * DenseLabels here are always real-valued and thus applicable to
	 * classification
	 * (cf.  CClassifier) and regression (cf. CRegression) problems.
	 *
	 * This class implements the shared functions for storing, and accessing
	 * label
	 * (vectors).
	 */
	class CDenseLabels : public CLabels
	{
	public:
		/** default constructor */
		CDenseLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		CDenseLabels(int32_t num_labels);

		/** copy constructor
		 *
		 * @param orig The dense labels to copy
		 */
		CDenseLabels(const CDenseLabels& orig);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		CDenseLabels(CFile* loader);

		/** destructor */
		virtual ~CDenseLabels();

		/** Make sure the label is valid, otherwise raise SG_ERROR.
		 *
		 * possible with subset
         *
         * @param context optional message to convey the context
		 */
		void ensure_valid(const char* context = NULL) override;

		/** load labels from file
		 *
		 * any subset is removed before
		 *
		 * @param loader File object via which to load data
		 */
		virtual void load(CFile* loader);

		/** save labels to file
		 *
		 * not possible with subset
		 *
		 * @param writer File object via which to save data
		 */
		virtual void save(CFile* writer);

		/** set label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to set
		 * @param label value of label
		 * @return if setting was successful
		 */
		bool set_label(int32_t idx, float64_t label);

		/** set INT label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to set
		 * @param label INT value of label
		 * @return if setting was successful
		 */
		bool set_int_label(int32_t idx, int32_t label);

		/** Getter for labels
		 *
		 * @return labels, a copy if a subset is set
		 */
		SGVector<float64_t> get_labels() const;

		/** get label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to get
		 * @return value of label
		 */
		float64_t get_label(int32_t idx) const;

		/** get label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to get
		 * @return value of label
		 */
		template<typename ST>
		ST get_label_t(int32_t idx)
		{
			REQUIRE(idx<get_num_labels(), "The provided index (%d) is out of bounds (the last label has index (%d)).  "
				"Please ensure that you're using a valid index number.", idx, get_num_labels())
			REQUIRE(m_labels.vector, "You're attempting to get a label when there are in fact none!  "
				"Please ensure that you initialized the labels correctly.")
			int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
			return m_labels.vector[real_num];
		}

		/** get INT label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to get
		 * @return INT value of label
		 */
		int32_t get_int_label(int32_t idx);

		/** Gets a copy of the labels.
		 *
		 * possible with subset
		 *
		 * @return labels
		 */
		SGVector<float64_t> get_labels_copy() const;

		/** Getter for labels
		 *
		 * @return labels, or a copy of them if a subset is
		 * set or if ST is not of type float64_t
		 */
		template<typename ST>
		SGVector<ST> get_labels_t()
		{
			if (m_subset_stack->has_subsets())
				return get_labels_copy_t<ST>();

			SGVector<ST> labels_copy(m_labels.vlen);
			for(index_t i = 0; i < m_labels.vlen; ++i)
				labels_copy[i] = (ST) m_labels[i];

			return labels_copy;
		}

		/** Get copy of labels.
		 *
		 * possible with subset
		 *
		 * @return labels
		 */
		template<typename ST>
		SGVector<ST> get_labels_copy_t()
		{
			if (!m_subset_stack->has_subsets())
			{
				SGVector<ST> labels_copy(m_labels.vlen);
				for(index_t i = 0; i < m_labels.vlen; ++i)
					labels_copy[i] = (ST) m_labels[i];

				return labels_copy;
			}

			index_t num_labels = get_num_labels();
			SGVector<ST> result(num_labels);

			/* copy element wise because of possible subset */
			for (index_t i=0; i<num_labels; i++)
				result[i] = get_label_t<ST>(i);

			return result;
		}

		/** set labels
		 *
		 * not possible with subset
		 *
		 * @param v labels
		 */
		void set_labels(SGVector<float64_t> v);

		/**
		 * set all labels to +1
		 *
		 * possible with subset
		 * */
		void set_to_one();

		/**
		 * set all labels to zero
		 *
		 * possible with subset
		 * */
		void zero();

		/**
		 * set all labels to a const value
		 *
		 * possible with subset
		 *
		 * @param c const to set labels to
		 * */
		void set_to_const(float64_t c);

		/** get INT label vector
		 *
		 * possible with subset
		 *
		 * @return INT labels
		 */
		SGVector<int32_t> get_int_labels();

		/** set INT labels
		 *
		 * not possible on subset
		 *
		 * @param labels INT labels
		 */
		void set_int_labels(SGVector<int32_t> labels);

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
		/** set INT64 labels
		 *
		 * not possible on subset
		 *
		 * @param labels INT labels
		 */
		void set_int_labels(SGVector<int64_t> labels);
#endif

		/** get number of labels, depending on whether a subset is set
		 *
		 * @return number of labels
		 */
		int32_t get_num_labels() const override;

		/** get label type
		 *
		 * @return label type (binary, multiclass, ...)
		 */
		ELabelType get_label_type() const override
		{
			return LT_DENSE_GENERIC;
		}

		const char* get_name() const override
		{
			return "DenseLabels";
		}

	public:
		/** label designates classify reject */
		static const int32_t REJECTION_LABEL = -2;

	private:
		void init();

	protected:
		/** the label vector */
		SGVector<float64_t> m_labels;
};

/**
* Specialization of get_labels for float64_t
* @return labels, or a copy of them if a subset
* is set
*/
template <>
inline SGVector<float64_t> CDenseLabels::get_labels_t<float64_t>()
{
	if (m_subset_stack->has_subsets())
		return get_labels_copy_t<float64_t>();

	return m_labels;
	}
}
#endif
