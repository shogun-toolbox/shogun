/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Yuyu Zhang
 */

#ifndef _SEQUENCE_LABELS__H__
#define _SEQUENCE_LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

class SequenceLabels;

/** @brief Class CSequence to be used in the application of Structured Output
 * (SO) learning to Hidden Markov Support Vector Machines (HM-SVM). */
class Sequence : public StructuredData
{
public:
	/** data type */
	STRUCTURED_DATA_TYPE(SDT_SEQUENCE);

	/** constructor
	 *
	 * @param seq data sequence
	 */
	Sequence(SGVector< int32_t > seq = SGVector<int32_t>()) : StructuredData(), data(seq) { }

	/** destructor */
	~Sequence() { }

	/** helper method used to specialize a base class instance
	 *
	 * @param base_data its dynamic type must be CSequence
	 */
	static std::shared_ptr<Sequence> obtain_from_generic(std::shared_ptr<StructuredData> base_data)
	{
		if ( base_data->get_structured_data_type() == SDT_SEQUENCE )
			return std::static_pointer_cast<Sequence>(base_data);
		else
			error("base_data must be of dynamic type CSequence");

		return NULL;
	}

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "Sequence"; }

	/** returns data */
	SGVector<int32_t> get_data() const { return data; }

protected:
	/** data sequence */
	SGVector< int32_t > data;

};

/** @brief Class CSequenceLabels used e.g. in the application of Structured Output
 * (SO) learning to Hidden Markov Support Vector Machines (HM-SVM). Each of the
 * labels is represented by a sequence of integers. Each label is of type
 * CSequence and all of them are stored in a DynamicObjectArray. */
class SequenceLabels : public StructuredLabels
{
	public:
		/** default constructor */
		SequenceLabels();

		/** standard constructor
		 *
		 * @param num_labels number of labels
		 * @param num_states number of states
		 */
		SequenceLabels(int32_t num_labels, int32_t num_states);

		/**
		 * constructor using the data of all the labels concatenated. All the
		 * labels are assumed to have the same length. The length of labels must
		 * be equal to label_length times num_labels.
		 *
		 * @param labels concatenation of the labels
		 * @param label_length number of elements in each label
		 * @param num_labels number of labels
		 * @param num_states number of states
		 */
		SequenceLabels(SGVector< int32_t > labels, int32_t label_length, int32_t num_labels, int32_t num_states);

		/** destructor */
		virtual ~SequenceLabels();

		/** @return object name */
		virtual const char* get_name() const { return "SequenceLabels"; }

		/**
		 * add a new label to the vector of labels, effectively
		 * increasing the number of elements of the structure. This
		 * method should be used when inserting labels for the first
		 * time. NOTE: the elements of the labels have to be in the
		 * interval [0, 1, ..., num_states-1].
		 *
		 * @param label label to add
		 */
		void add_vector_label(SGVector< int32_t > label);

		/** get the number of states
		 *
		 * @return the number of states
		 */
		inline int32_t get_num_states() const { return m_num_states; };

	private:
		/** internal initialization */
		void init();

	private:
		/**
		 * the number of possible values taken by the elements of
		 * the sequences
		 */
		int32_t m_num_states;

}; /* CSequenceLabels */

} /* namespace shogun */

#endif /* _SEQUENCE_LABELS__H__ */
