/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Devramx
 */

#ifndef _LEVENSTEINDISTANCE_H___
#define _LEVENSTEINDISTANCE_H___

#include <shogun/distance/LevensteinDistance.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

namespace shogun
{
	constexpr uint8_t AdditionCost = 1;
	constexpr uint8_t DeletionCost = 1;
	constexpr uint8_t MutationCost = 2;
	size_t Levenstein( const std::string &lhs, const std::string &rhs);
} // namespace shogun
#endif /*_LEVENSTEINDISTANCE_H___*/