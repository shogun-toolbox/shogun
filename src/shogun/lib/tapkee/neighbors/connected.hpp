/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
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
