/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Abinash Panda, Yuyu Zhang, Shell Hu
 */

#ifndef _STRUCTURED_DATA_TYPES__H__
#define _STRUCTURED_DATA_TYPES__H__

#include <shogun/lib/config.h>

namespace shogun
{

/** structured data type */
enum EStructuredDataType
{
	SDT_UNKNOWN = 0,
	SDT_REAL = 1,
	SDT_SEQUENCE = 2,
	SDT_FACTOR_GRAPH = 3,
	SDT_SPARSE_MULTILABEL = 4
};

#define STRUCTURED_DATA_TYPE(SDT) \
	/** returns structured data type used by the labels \
	 *
	 * @return structured data type
	 */ \
	virtual EStructuredDataType get_structured_data_type() const { return SDT; }
}

#endif /* _STRUCTURED_DATA_TYPES__H__ */
