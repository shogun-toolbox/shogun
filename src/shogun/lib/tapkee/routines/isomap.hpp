/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 *
 */

#ifndef TAPKEE_ISOMAP_H_
#define TAPKEE_ISOMAP_H_

#include <shogun/lib/tapkee/tapkee_defines.hpp>
#include <shogun/lib/tapkee/utils/fibonacci_heap.hpp>
#include <shogun/lib/tapkee/utils/time.hpp>
#include <limits>

using std::numeric_limits;

namespace tapkee
{
namespace tapkee_internal
{

//! Computes shortest distances (so-called geodesic distances)
//! using Dijkstra algorithm.
//!
//! @param begin begin data iterator
//! @param end end data iterator
//! @param neighbors neighbors of each vector
//! @param callback distance callback
//!
template <class RandomAccessIterator, class DistanceCallback>
DenseSymmetricMatrix compute_shortest_distances_matrix(const RandomAccessIterator& begin, const RandomAccessIterator& end, 
		const Neighbors& neighbors, const DistanceCallback& callback)
{
	timed_context context("Distances shortest path relaxing");
	const IndexType n_neighbors = neighbors[0].size();
	const IndexType N = (end-begin);

	DenseSymmetricMatrix shortest_distances(N,N);
	
#pragma omp parallel shared(shortest_distances,neighbors,begin,callback) default(none)
	{
		bool* f = new bool[N];
		bool* s = new bool[N];
		IndexType k;
		FibonacciHeap* heap = new FibonacciHeap(N);

#pragma omp for nowait
		for (k=0; k<N; k++)
		{
			// fill s and f with false, fill shortest_D with infinity
			for (IndexType j=0; j<N; j++)
			{
				shortest_distances(k,j) = numeric_limits<DenseMatrix::Scalar>::max();
				s[j] = false;
				f[j] = false;
			}
			// set distance from k to k as zero
			shortest_distances(k,k) = 0.0;

			// insert kth object to heap with zero distance and set f[k] true
			heap->insert(k,0.0);
			f[k] = true;

			// while heap is not empty
			while (heap->get_num_nodes()>0)
			{
				// extract min and set (s)olution state as true and (f)rontier as false
				ScalarType tmp;
				int min_item = heap->extract_min(tmp);
				s[min_item] = true;
				f[min_item] = false;

				// for-each edge (min_item->w)
				for (IndexType i=0; i<n_neighbors; i++)
				{
					// get w idx
					int w = neighbors[min_item][i];
					// if w is not in solution yet
					if (s[w] == false)
					{
						// get distance from k to i through min_item
						ScalarType dist = shortest_distances(k,min_item) + callback(begin[min_item],begin[w]);
						// if distance can be relaxed
						if (dist < shortest_distances(k,w))
						{
							// relax distance
							shortest_distances(k,w) = dist;
							// if w is in (f)rontier
							if (f[w])
							{
								// decrease distance in heap
								heap->decrease_key(w, dist);
							}
							else
							{
								// insert w to heap and set (f)rontier as true
								heap->insert(w, dist);
								f[w] = true;
							}
						}
					}
				}
			}
			heap->clear();
		}

		delete heap;
		delete[] s;
		delete[] f;
	}
	return shortest_distances;
}

//! Computes shortest distances (so-called geodesic distances)
//! using Dijkstra algorithm with landmarks.
//!
//! @param begin begin data iterator
//! @param end end data iterator
//! @param landmarks landmarks
//! @param neighbors neighbors of each vector
//! @param callback distance callback
//!
template <class RandomAccessIterator, class DistanceCallback>
DenseMatrix compute_shortest_distances_matrix(const RandomAccessIterator& begin, const RandomAccessIterator& end, 
		const Landmarks& landmarks, const Neighbors& neighbors, const DistanceCallback& callback)
{
	timed_context context("Distances shortest path relaxing");
	const IndexType n_neighbors = neighbors[0].size();
	const IndexType N = end-begin;

	DenseMatrix shortest_distances(landmarks.size(),N);
	
#pragma omp parallel shared(shortest_distances,begin,landmarks,neighbors,callback) default(none)
	{
		bool* f = new bool[N];
		bool* s = new bool[N];
		IndexType k;
		FibonacciHeap* heap = new FibonacciHeap(N);

#pragma omp for nowait
		for (k=0; k<landmarks.size(); k++)
		{
			// fill s and f with false, fill shortest_D with infinity
			for (IndexType j=0; j<N; j++)
			{
				shortest_distances(k,j) = numeric_limits<DenseMatrix::Scalar>::max();
				s[j] = false;
				f[j] = false;
			}
			// set distance from k to k as zero
			shortest_distances(k,landmarks[k]) = 0.0;

			// insert kth object to heap with zero distance and set f[k] true
			heap->insert(landmarks[k],0.0);
			f[k] = true;

			// while heap is not empty
			while (heap->get_num_nodes()>0)
			{
				// extract min and set (s)olution state as true and (f)rontier as false
				ScalarType tmp;
				int min_item = heap->extract_min(tmp);
				s[min_item] = true;
				f[min_item] = false;

				// for-each edge (min_item->w)
				for (IndexType i=0; i<n_neighbors; i++)
				{
					// get w idx
					int w = neighbors[min_item][i];
					// if w is not in solution yet
					if (s[w] == false)
					{
						// get distance from k to i through min_item
						ScalarType dist = shortest_distances(k,min_item) + callback(begin[min_item],begin[w]);
						// if distance can be relaxed
						if (dist < shortest_distances(k,w))
						{
							// relax distance
							shortest_distances(k,w) = dist;
							// if w is in (f)rontier
							if (f[w])
							{
								// decrease distance in heap
								heap->decrease_key(w, dist);
							}
							else
							{
								// insert w to heap and set (f)rontier as true
								heap->insert(w, dist);
								f[w] = true;
							}
						}
					}
				}
			}
			heap->clear();
		}
	
		delete heap;
		delete[] s;
		delete[] f;
	}
	return shortest_distances;
}

}
}

#endif
