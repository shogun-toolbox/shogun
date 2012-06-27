/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _HMSVM_LABELS__H__
#define _HMSVM_LABELS__H__

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

class CHMSVMLabels;

/** @brief Class CSequence to be used in the application of Structured Output
 * (SO) learning to Hidden Markov Support Vector Machines (HM-SVM). */
struct CSequence : public CStructuredData
{
	STRUCTURED_DATA_TYPE(SDT_SEQUENCE);

	/** constructor
	 *
	 * @param seq data sequence
	 */
	CSequence(SGVector< int32_t > seq) : CStructuredData(), data(seq) { }

	/** helper method used to specialize a base class instance
	 *
	 * @param base_data its dynamic type must be CSequence
	 */
	static CSequence* obtain_from_generic(CStructuredData* base_data)
	{
		if ( base_data->get_structured_data_type() == SDT_SEQUENCE )
			return (CSequence*) base_data;
		else
			SG_SERROR("base_data must be of dynamic type CSequence\n");

		return NULL;
	}

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "Sequence"; }

	/** data sequence */
	SGVector< int32_t > data;
};

/** @brief Class CHMSVMLabels to be used in the application of Structured Output
 * (SO) learning to Hidden Markov Support Vector Machines (HM-SVM). Each of the
 * labels is represented by a sequence of integers. Each label is of type
 * CSequence and all of them are stored in a CDynamicObjectArray. */
class CHMSVMLabels : public CStructuredLabels
{
	public:
		/** default constructor */
		CHMSVMLabels();

		/** constructor
		 *
		 * @param src labels to set
		 */
		CHMSVMLabels(SGMatrix< int32_t > const src);

		/** destructor */
		virtual ~CHMSVMLabels();

		/** @return object name */
		virtual const char* get_name() const { return "HMSVMLabels"; }

}; /* CHMSVMLabels */

} /* namespace shogun */

#endif /* _HMSVM_LABELS__H__ */
