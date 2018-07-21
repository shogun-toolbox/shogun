/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */
 
#include <string.h> 
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/features/FeatureTypes.h>

namespace shogun
{
	std::string feature_type(EFeatureType f)
	{
		switch (f)
		{
		case F_BOOL:
			return "BOOL";
		case F_CHAR:
			return "CHAR";
		case F_BYTE:
			return "BYTE";
		case F_SHORT:
			return "SHORT";
		case F_WORD:
			return "WORD";
		case F_INT:
			return "INT";
		case F_UINT:
			return "UINT";
		case F_LONG:
			return "LONG";
		case F_ULONG:
			return "ULONG";
		case F_SHORTREAL:
			return "SHORTREAL";
		case F_DREAL:
			return "DREAL";
		case F_LONGREAL:
			return "LONGREAL";
		case F_ANY:
			return "ANY";
		default:
			SG_SNOTIMPLEMENTED
			return "UNKNOWN";
		}
	}
}
