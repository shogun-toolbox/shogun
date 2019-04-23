/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Soeren Sonnenburg, Thoralf Klein, Yuyu Zhang,
 *          Bjoern Esser, Sergey Lisitsyn
 */

#ifndef _MULTICLASS_SO_LABELS__H__
#define _MULTICLASS_SO_LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

class StructuredLabels;
class MulticlassSOLabels;

/** @brief Class RealNumber to be used in the application of Structured
 * Output (SO) learning to multiclass classification. Even though it is likely
 * that it does not make sense to consider real numbers as structured data,
 * it has been made in this way because the basic type to use in structured
 * labels needs to inherit from StructuredData. */
struct RealNumber : public StructuredData
{
	/** data type */
	STRUCTURED_DATA_TYPE(SDT_REAL);

	/** constructor
	 *
	 * @param val value of the real number
	 */
	RealNumber(float64_t val) : StructuredData(), value(val) { }

	/** helper method used to specialize a base class instance
	 *
	 * @param base_data its dynamic type must be RealNumber
	 */
	static std::shared_ptr<RealNumber> obtain_from_generic(std::shared_ptr<StructuredData> base_data)
	{
		if ( base_data->get_structured_data_type() == SDT_REAL )
			return std::static_pointer_cast<RealNumber>(base_data);
		else
			error("base_data must be of dynamic type RealNumber");

		return NULL;
	}

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "RealNumber"; }

	/** value of the real number */
	float64_t value;
};

/** @brief Class CMulticlassSOLabels to be used in the application of Structured
 * Output (SO) learning to multiclass classification. Each of the labels is
 * represented by a real number and it is required that the values of the labels
 * are in the set {0, 1, ..., num_classes-1}. Each label is of type RealNumber
 * and all of them are stored in a DynamicObjectArray. */
class MulticlassSOLabels : public StructuredLabels
{
	public:
		/** default constructor */
		MulticlassSOLabels();

		/** create label vector of given size
		 *
		 * @param num_labels size of label vector
		 */
		MulticlassSOLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param src labels to set
		 */
		MulticlassSOLabels(SGVector< float64_t > const src);

		/** destructor */
		virtual ~MulticlassSOLabels();

		/** get number of classes
		 *
		 * @return number of classes
		 */
		inline int32_t get_num_classes() { return m_num_classes; }

		/**
		 * add a new label to the vector of labels, effectively
		 * increasing the number of elements of the structure. This
		 * method should be used when inserting labels for the first
		 * time.
		 *
		 * @param label label to add
		 */
		virtual void add_label(std::shared_ptr<StructuredData> label);

		/** get label object for specified index
		 *
		 * @param idx index of the label
		 *
		 * @return label object
		 */
		virtual std::shared_ptr<StructuredData> get_label(int32_t idx);

		/**
		 * set label, possible with subset. This method should be used
		 * when substituting labels previously inserted. To insert new
		 * labels, use the method add_label.
		 *
		 * @param idx index of label to set
		 * @param label value of label
		 *
		 * @return if setting was successful
		 */
		virtual bool set_label(int32_t idx, std::shared_ptr<StructuredData> label);

		/** get number of labels, depending on wheter a subset is set
		 *
		 * @return number of labels
		 */
		virtual int32_t get_num_labels() const;

		/** @return object name */
		virtual const char* get_name() const { return "MulticlassSOLabels"; }

	private:
		void init();

	private:
		/** number of classes */
		int32_t m_num_classes;

		SGVector< float64_t > m_labels_vector;
		int32_t m_num_labels_set;

}; /* CMulticlassSOLabels */

} /* namespace shogun */

#endif /* _MULTICLASS_SO_LABELS__H__ */
