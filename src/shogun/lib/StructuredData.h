/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Yuyu Zhang, Björn Esser
 */

#ifndef _STRUCTUREDDATA_H__
#define _STRUCTUREDDATA_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

/** @brief Base class of the components of StructuredLabels */
class CStructuredData : public CSGObject
{

	public:
		/** default constructor */
		CStructuredData();

		/** destructor */
		virtual ~CStructuredData();

		/** @return object name */
		virtual const char* get_name() const { return "StructuredData"; }

		/** get structured data type
		 *
		 * @return structured data type (a sequence, a tree, a graph, ...)
		 */
		virtual EStructuredDataType get_structured_data_type() const;

};

} /* namespace shogun */

#endif /* _STRUCTUREDDATA_H__ */
