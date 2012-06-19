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

namespace shogun {

class CStructuredLabels : public CLabels
{

	public:
		/** default constructor */
		CStructuredLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		CStructuredLabels(int32_t num_labels);

		/** destructor */
		virtual ~CStructuredLabels();

		/** helper method used to specialize a base class instance
		 *
		 * @param base_labels its dynamic type must be CStructuredLabels
		 */
		static CStructuredLabels* obtain_from_generic(CLabels* base_labels);

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
		void add_label(CStructuredData* label);

		/** get labels
		 *
		 * not possible with subset
		 *
		 * @return labels
		 */
		CDynamicObjectArray* get_labels() const;
		
		/** get label object for specified index
		 *
		 * @param idx index of tha label
		 *
		 * @return label object
		 */
		CStructuredData* get_label(int32_t idx);

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
		bool set_label(int32_t idx, CStructuredData* label);

		/** get number of labels, depending on wheter a subset is set
		 *
		 * @return number of labels
		 */
		virtual int32_t get_num_labels();

		/** @return object name */
		virtual const char* get_name() const { return "StructuredLabels"; }

		/** get label type
		 *
		 * @return label type LT_STRUCTURED
		 */
		virtual ELabelType get_label_type() { return LT_STRUCTURED; }

	private:
		/** internal initialization */
		void init();

	protected:
		/** the vector of labels */
		CDynamicObjectArray* m_labels;

}; /* class CStructuredLabels */

} /* namespace shogun */

#endif /* _STRUCTUREDLABELS_H__ */
