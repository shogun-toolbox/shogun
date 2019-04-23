/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Thoralf Klein
 */

#include <shogun/lib/StructuredData.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/StructuredDataTypes.h>

using namespace shogun;

StructuredData::StructuredData()
: SGObject()
{
}

StructuredData::~StructuredData()
{
}

EStructuredDataType StructuredData::get_structured_data_type() const
{
	SG_ERROR("get_structured_data_type() not defined. \n"
		 "Make sure that STRUCTURED_DATA_TYPE(SDT) is defined "
		 "in every class that inherits from StructuredData.\n");

	return SDT_UNKNOWN;
};
