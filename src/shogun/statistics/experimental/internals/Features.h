/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2014  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __FEATURES_H_
#define __FEATURES_H_

#include <shogun/lib/config.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>

namespace shogun
{

namespace internal
{

// feat traits - required for proper typecasting
template <EFeatureClass C, EFeatureType T>
struct feats_traits
{
	using type = CFeatures;
};

template <>
struct feats_traits<EFeatureClass::C_DENSE, EFeatureType::F_DREAL>
{
	using type = CDenseFeatures<float64_t>;
};

template <>
struct feats_traits<EFeatureClass::C_STREAMING_DENSE, EFeatureType::F_DREAL>
{
	using type = CStreamingDenseFeatures<float64_t>;
};

}

}

#endif // __FEATURES_H_
