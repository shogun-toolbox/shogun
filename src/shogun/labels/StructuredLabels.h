/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTURED_LABELS__H__
#define _STRUCTURED_LABELS__H__

#include <shogun/labels/Labels.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun {

/** @brief Base class of the labels used in Structured Output (SO) problems */
class CStructuredLabels : public CLabels
{

	public:
		/** default constructor */
		CStructuredLabels();

		/** constructor
		 *
		 * This method reserves memory to store num_labels without the
		 * need of allocating more memory in the future when inserting
		 * labels with other method, e.g. add_label.
		 *
		 * @param num_labels number of labels to pre-allocate
		 */
		CStructuredLabels(int32_t num_labels);

		/** destructor */
		virtual ~CStructuredLabels();

		/** check if labeling is valid
		 *
		 * possible with subset
		 *
		 * @return if labeling is valid
		 */
		virtual void ensure_valid(const char* context = NULL);

		/**
		 * add a new label to the vector of labels, effectively
		 * increasing the number of elements of the structure. This
		 * method should be used when inserting labels for the first
		 * time.
		 *
		 * @param label label to add
		 */
		virtual void add_label(CStructuredData* label);

		/** get labels
		 *
		 * not possible with subset
		 *
		 * @return labels
		 */
		CDynamicObjectArray* get_labels() const;

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
		virtual const char* get_name() const { return "StructuredLabels"; }

		/** get label type
		 *
		 * @return label type LT_STRUCTURED
		 */
		virtual ELabelType get_label_type() const { return LT_STRUCTURED; }

		/** get structured data type the labels are composed of
		 *
		 * @return structured data type
		 */
		inline EStructuredDataType get_structured_data_type() { return m_sdt; }

	private:
		/** internal initialization */
		void init();

		/** ensure that the correct structured data type is used */
		void ensure_valid_sdt(CStructuredData* label);

	protected:
		/** the vector of labels */
		CDynamicObjectArray* m_labels;

		/** the structured data type the labels are composed of */
		EStructuredDataType m_sdt;

}; /* class CStructuredLabels */

} /* namespace shogun */

#endif /* _STRUCTUREDLABELS_H__ */
