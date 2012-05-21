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
		~CStructuredLabels();

		/** check if labeling is valid 
		 *
		 * possible with subset
		 *
		 * @return if labeling is valid
		 */
		virtual bool is_valid();

		/** get labels
		 *
		 * not possible with subset
		 *
		 * @return labels
		 */
		CDynamicObjectArray* get_labels() const;

		/** get number of labels, depending on whether a subset is set
		 *
		 * @return number of labels
		 */
		virtual int32_t get_num_labels();

		/** @return object name */
		inline virtual const char* get_name() const { return "StructuredLabels"; }

		/** get label type
		 *
		 * @return label type LT_STRUCTURED 
		 */
		virtual ELabelType get_label_type() { return LT_STRUCTURED; }
	private:
		/** internal initialization */
		void init();

	private:
		/** the vector of labels */
		CDynamicObjectArray* m_labels;

}; /* class CStructuredLabels */

} /* namespace shogun */

#endif /* _STRUCTURED_LABELS__H__ */
