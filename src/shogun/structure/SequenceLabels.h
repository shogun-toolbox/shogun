/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Fernando José Iglesias García
 * Copyright (C) 2012-2013 Fernando José Iglesias García
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

class CSequenceLabels;

/** @brief Class CSequence to be used in the application of Structured Output
 * (SO) learning to Hidden Markov Support Vector Machines (HM-SVM). */
class CSequence : public CStructuredData
{
public:
	/** data type */
	STRUCTURED_DATA_TYPE(SDT_SEQUENCE);

	/** constructor
	 *
	 * @param seq data sequence
	 */
	CSequence(SGVector< int32_t > seq = SGVector<int32_t>()) : CStructuredData(), data(seq) { }

	/** destructor */
	~CSequence() { }

	/** helper method used to specialize a base class instance
	 *
	 * @param base_data its dynamic type must be CSequence
	 */
	static CSequence* obtain_from_generic(CStructuredData* base_data)
	{
		if ( base_data->get_structured_data_type() == SDT_SEQUENCE )
			return (CSequence*) base_data;
		else
			SG_SERROR("base_data must be of dynamic type CSequence\n")

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
 * CSequence and all of them are stored in a CDynamicObjectArray. */
class CSequenceLabels : public CStructuredLabels
{
	public:
		/** default constructor */
		CSequenceLabels();

		/** standard constructor
		 *
		 * @param num_labels number of labels
		 * @param num_states number of states
		 */
		CSequenceLabels(int32_t num_labels, int32_t num_states);

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
		CSequenceLabels(SGVector< int32_t > labels, int32_t label_length, int32_t num_labels, int32_t num_states);

		/** destructor */
		virtual ~CSequenceLabels();

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
