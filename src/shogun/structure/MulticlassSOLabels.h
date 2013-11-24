/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _MULTICLASS_SO_LABELS__H__
#define _MULTICLASS_SO_LABELS__H__

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

class CMulticlassSOLabels;

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
	static RealNumber* obtain_from_generic(StructuredData* base_data)
	{
		if ( base_data->get_structured_data_type() == SDT_REAL )
			return (RealNumber*) base_data;
		else
			SG_SERROR("base_data must be of dynamic type RealNumber\n")

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
 * and all of them are stored in a CDynamicObjectArray. */
class CMulticlassSOLabels : public CStructuredLabels
{
	public:
		/** default constructor */
		CMulticlassSOLabels();

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

		/** @return object name */
		virtual const char* get_name() const { return "MulticlassSOLabels"; }

	private:
		void init();

	private:
		/** number of classes */
		int32_t m_num_classes;

}; /* CMulticlassSOLabels */

} /* namespace shogun */

#endif /* _MULTICLASS_SO_LABELS__H__ */
