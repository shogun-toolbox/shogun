/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Shashwat Lal Das, Viktor Gal, 
 *          Fernando Iglesias, Sergey Lisitsyn, Shell Hu, Liang Pang, Wu Lin
 */
#ifndef _FEATURE_TYPES__H__
#define _FEATURE_TYPES__H__

#include <shogun/lib/config.h>

namespace shogun
{

	/// shogun feature type
	enum EFeatureType
	{
		F_UNKNOWN = 0,
		F_BOOL = 5,
		F_CHAR = 10,
		F_BYTE = 20,
		F_SHORT = 30,
		F_WORD = 40,
		F_INT = 50,
		F_UINT = 60,
		F_LONG = 70,
		F_ULONG = 80,
		F_SHORTREAL = 90,
		F_DREAL = 100,
		F_LONGREAL = 110,
		F_ANY = 1000
	};

	/// shogun feature class
	enum EFeatureClass
	{
		C_UNKNOWN = 0,
		C_DENSE = 10,
		C_SPARSE = 20,
		C_STRING = 30,
		C_COMBINED = 40,
		C_COMBINED_DOT = 60,
		C_WD = 70,
		C_SPEC = 80,
		C_WEIGHTEDSPEC = 90,
		C_POLY = 100,
		C_STREAMING_DENSE = 110,
		C_STREAMING_SPARSE = 120,
		C_STREAMING_STRING = 130,
		C_STREAMING_VW = 140,
		C_BINNED_DOT = 150,
		C_DIRECTOR_DOT = 160,
		C_LATENT = 170,
		C_MATRIX = 180,
		C_FACTOR_GRAPH = 190,
		C_INDEX = 200,
		C_SUB_SAMPLES_DENSE=300,
		C_ANY = 1000
	};

	/// shogun feature properties
	enum EFeatureProperty
	{
		FP_NONE = 0,
		FP_DOT = 1,
		FP_STREAMING_DOT = 2
	};
	std::string feature_type(EFeatureType f);
}
#endif // _FEATURE_TYPES__H__
