/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
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
 * along with this program.  If not, see <http:/www.gnu.org/licenses/>.
 */

#include <shogun/features/Features.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/hypothsistest/internals/DataFetcher.h>
#include <shogun/hypothsistest/internals/StreamingDataFetcher.h>
#include <shogun/hypothsistest/internals/DataFetcherFactory.h>

using namespace shogun;
using namespace internal;

DataFetcher* DataFetcherFactory::get_instance(CFeatures* feats)
{
	EFeatureClass fclass = feats->get_feature_class();
	if (fclass == C_STREAMING_DENSE || fclass == C_STREAMING_SPARSE || fclass == C_STREAMING_STRING)
	{
		return new StreamingDataFetcher(static_cast<CStreamingFeatures*>(feats));
	}
	return new DataFetcher(feats);
}

