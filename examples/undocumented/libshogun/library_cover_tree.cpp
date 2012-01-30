/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/CoverTree.h>

using namespace shogun;

class TEST_COVERTREE_POINT
{
public:

	TEST_COVERTREE_POINT(int32_t index, double value)
	{
		point_index = index;
		point_value = value;
	}

	inline double distance(const TEST_COVERTREE_POINT& p) const
	{
		return CMath::abs(p.point_value-point_value);
	}

	inline bool operator==(const TEST_COVERTREE_POINT& p) const
	{
		return (p.point_index==point_index);
	}

	int point_index;
	double point_value;
};


int main(int argc, char** argv)
{
	init_shogun();

	int N = 100;
	CoverTree<TEST_COVERTREE_POINT> coverTree(N);
	for (int i=0; i<N; i++)
		coverTree.insert(TEST_COVERTREE_POINT(i,i*i));
	std::vector<TEST_COVERTREE_POINT> neighbors = 
	   coverTree.kNearestNeighbors(TEST_COVERTREE_POINT(0,0.0),N-1);

	exit_shogun();
	return 0;
}
