/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Thoralf Klein
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _MULTICLASS_SO_LABELS__H__
#define _MULTICLASS_SO_LABELS__H__

#include <labels/StructuredLabels.h>
#include <lib/SGVector.h>
#include <lib/StructuredData.h>
#include <lib/StructuredDataTypes.h>

namespace shogun
{

class CMulticlassSOLabels;

/** @brief Class CRealNumber to be used in the application of Structured
 * Output (SO) learning to multiclass classification. Even though it is likely
 * that it does not make sense to consider real numbers as structured data,
 * it has been made in this way because the basic type to use in structured
 * labels needs to inherit from CStructuredData. */
struct CRealNumber : public CStructuredData
{
	/** data type */
	STRUCTURED_DATA_TYPE(SDT_REAL);

	/** constructor
	 *
	 * @param val value of the real number
	 */
	CRealNumber(float64_t val) : CStructuredData(), value(val) { }

	/** helper method used to specialize a base class instance
	 *
	 * @param base_data its dynamic type must be CRealNumber
	 */
	static CRealNumber* obtain_from_generic(CStructuredData* base_data)
	{
		if ( base_data->get_structured_data_type() == SDT_REAL )
			return (CRealNumber*) base_data;
		else
			SG_SERROR("base_data must be of dynamic type CRealNumber\n")

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
 * are in the set {0, 1, ..., num_classes-1}. Each label is of type CRealNumber
 * and all of them are stored in a CDynamicObjectArray. */
class CMulticlassSOLabels : public CStructuredLabels
{
	public:
		/** default constructor */
		CMulticlassSOLabels();

		CMulticlassSOLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param src labels to set
		 */
		CMulticlassSOLabels(SGVector< float64_t > const src);

		/** destructor */
		virtual ~CMulticlassSOLabels();

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
		virtual void add_label(CStructuredData* label);

		/** get label object for specified index
		 *
		 * @param idx index of the label
		 *
		 * @return label object
		 */
		virtual CStructuredData* get_label(int32_t idx);

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
		virtual bool set_label(int32_t idx, CStructuredData* label);

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
