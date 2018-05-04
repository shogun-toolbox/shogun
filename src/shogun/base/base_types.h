/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#ifndef BASE_TYPES_H
#define BASE_TYPES_H

namespace shogun
{

	// all shogun base classes for put/add templates
	class CMachine;
	class CKernel;
	class CDistance;
	class CFeatures;
	class CLabels;
	class CECOCEncoder;
	class CECOCDecoder;
	class CMulticlassStrategy;

	// type trait to enable certain methods only for shogun base types
	template <class T>
	struct is_sg_base
	    : std::integral_constant<
	          bool, std::is_same<CMachine, T>::value ||
	                    std::is_same<CKernel, T>::value ||
	                    std::is_same<CDistance, T>::value ||
	                    std::is_same<CFeatures, T>::value ||
	                    std::is_same<CLabels, T>::value ||
	                    std::is_same<CECOCEncoder, T>::value ||
	                    std::is_same<CECOCDecoder, T>::value ||
	                    std::is_same<CMulticlassStrategy, T>::value>
	{
	};
}

#endif // BASE_TYPES__H
