/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Saurabh Mahindre,
 *          Sergey Lisitsyn, Evgeniy Andreev, Yuyu Zhang, Chiyuan Zhang,
 *          Thoralf Klein, Fernando Iglesias, Leon Kuchenbecker
 */

#ifndef _LABELS__H__
#define _LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/features/SubsetStack.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>

namespace shogun
{
	class BinaryLabels;

	/** @brief The class Labels models labels, i.e. class assignments of
	 * objects.
	 *
	 * Labels is the base class for general Label objects.
	 *
	 * Label objects need to overload get_num_labels() and is_valid()
	 *
	 * (Multiple) Subsets (of subsets) are (partly) supported.
	 * Sub-classes may want to overwrite the subset_changed_post() method which
	 * is
	 * called automatically after each subset change. See method documentations
	 * to
	 * see how behaviour is changed when subsets are active.
	 * A subset is put onto a stack using the add_subset() method. The last
	 * added
	 * subset may be removed via remove_subset(). There is also the possibility
	 * to
	 * add subsets in place (this only stores one index vector in memory as
	 * opposed
	 * to many when add_subset() is called many times) with
	 * add_subset_in_place().
	 * The latter does not allow to remove such modifications one-by-one.
	 */
	class Labels : public SGObject
	{
	public:
		/** default constructor */
		Labels();

		/** copy constructor */
		Labels(const Labels& orig);

		/** destructor */
		virtual ~Labels();

		virtual bool is_valid() const = 0;

		/** Make sure the label is valid, otherwise raise SG_ERROR.
		 *
		 * possible with subset
		*
		* @param context optional message to convey the context
		 */
		virtual void ensure_valid(const char* context = NULL) = 0;

		/** get number of labels, depending on whether a subset is set
		 *
		 * @return number of labels
		 */
		virtual int32_t get_num_labels() const = 0;

		/** get label type
		 *
		 * @return label type (binary, multiclass, ...)
		 */
		virtual ELabelType get_label_type() const = 0;

		/** Adds a subset of indices on top of the current subsets (possibly
		 * subset of subset). Every call causes a new active index vector
		 * to be stored. Added subsets can be removed one-by-one. If this is not
		 * needed, add_subset_in_place() should be used (does not store
		 * intermediate index vectors)
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_subset(SGVector<index_t> subset);

		/** Sets/changes latest added subset. This allows to add multiple
		 * subsets
		 * with in-place memory requirements. They cannot be removed one-by-one
		 * afterwards, only the latest active can. If this is needed, use
		 * add_subset(). If no subset is active, this just adds.
		 *
		 * @param subset subset of indices to replace the latest one with.
		 * */
		virtual void add_subset_in_place(SGVector<index_t> subset);

		/** removes that last added subset from subset stack, if existing
		 * Calls subset_changed_post() afterwards */
		virtual void remove_subset();

		/** removes all subsets
		 * Calls subset_changed_post() afterwards */
		virtual void remove_all_subsets();

		/**
		 * @return subset stack
		 */
		virtual std::shared_ptr<SubsetStack> get_subset_stack();

		/** set the confidence value for a particular label
		 *
		 * @param value value to set
		 * @param idx label index whose conf. value is to be changed
		 */
		virtual void set_value(float64_t value, int32_t idx);

		/** get confidence value for a particular label
		 *
		 * @param idx label index
		 * @return confidence value of label with index idx
		 */
		virtual float64_t get_value(int32_t idx);

		/** set confidence vector
		 *
		 * @param values to be set (should have zero length to disable
		 * values or length must match the number of labels)
		 */
		virtual void set_values(SGVector<float64_t> values);

		/** get confidence vector
		 *
		 * @return confidences
		 */
		virtual SGVector<float64_t> get_values() const;

		/** shallow-copy of the labels object
		 *
		 * @return labels object
		 */
		virtual std::shared_ptr<Labels> duplicate() const
		{
			not_implemented(SOURCE_LOCATION);;
			return nullptr;
		}

#ifndef SWIG // SWIG should skip this part
		virtual std::shared_ptr<Labels> shallow_subset_copy()
		{
			not_implemented(SOURCE_LOCATION);;
			return NULL;
		}
#endif

	private:
		void init();

	protected:
		/** subset class to enable subset support for this class */
		std::shared_ptr<SubsetStack> m_subset_stack;

		/** current active value vector */
		SGVector<float64_t> m_current_values;
};
}
#endif
