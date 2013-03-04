/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (w) 2012-2013 Fernando J. Iglesias Garcia
 * Copyright (c) 2012-2013 Fernando J. Iglesias Garcia
 */

#ifndef TAPKEE_CONNECTED_H_
#define TAPKEE_CONNECTED_H_

#include <stack>
#include <vector>

namespace tapkee
{
namespace tapkee_internal
{

template <class RandomAccessIterator>
bool is_connected(RandomAccessIterator begin, RandomAccessIterator end,
		const Neighbors& neighbors)
{
	timed_context context("Checking if graph is connected");

	// The number of data points
	int N = end-begin;
	// The number of neighbors used in KNN
	IndexType k = neighbors[0].size();

	typedef std::stack<int> DFSStack;
	typedef std::vector<bool> VisitedVector;

	VisitedVector visited(N, false);
	DFSStack stack;
	int nvisited = 0;
	stack.push(0);

	while (!stack.empty())
	{
		int current = stack.top();
		stack.pop();

		if (visited[current])
			continue;

		visited[current] = true;
		++nvisited;
		
		if (nvisited == N) break;

		const LocalNeighbors& current_neighbors = neighbors[current];

		for(IndexType j=0; j<k; ++j)
		{
			int neighbor = current_neighbors[j];
			if (!visited[neighbor])
				stack.push(neighbor);
		}
	}

	return (nvisited==N);
}

} /* namespace tapkee_internal */
} /* namespace tapkee */

#endif /* TAPKEE_CONNECTED_H_ */
