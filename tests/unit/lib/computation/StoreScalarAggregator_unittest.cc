/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */
 
#include <shogun/lib/common.h>
#include <shogun/lib/computation/jobresult/ScalarResult.h>
#include <shogun/lib/computation/aggregator/StoreScalarAggregator.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(StoreScalarAggregator, submit_result)
{
	CStoreScalarAggregator<float64_t>* agg=new CStoreScalarAggregator<float64_t>;
	SG_REF(agg);
	const index_t num_results=10;

	for (index_t i=0; i<num_results; ++i)
	{
		CScalarResult<float64_t>* result
			=new CScalarResult<float64_t>((float64_t)i);
		SG_REF(result);
		agg->submit_result((CJobResult*)result);
		SG_UNREF(result);
	}
	agg->finalize();
	float64_t result=dynamic_cast<CScalarResult<float64_t>*>
		(agg->get_final_result())->get_result();

	EXPECT_NEAR(result, 45.0, 1E-16);
	SG_UNREF(agg);
}

