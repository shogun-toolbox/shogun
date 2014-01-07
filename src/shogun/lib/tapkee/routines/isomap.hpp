/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_Isomap_H_
#define TAPKEE_Isomap_H_

/* Tapkee includes */
#include <lib/tapkee/defines.hpp>
#include <lib/tapkee/utils/fibonacci_heap.hpp>
#include <lib/tapkee/utils/reservable_priority_queue.hpp>
#include <lib/tapkee/utils/time.hpp>
/* End of Tapkee includes */

#include <limits>

namespace tapkee
{
namespace tapkee_internal
{

#ifdef TAPKEE_USE_PRIORITY_QUEUE
typedef std::pair<IndexType,ScalarType> HeapElement;

struct HeapElementComparator
{
	inline bool operator()(const HeapElement& l, const HeapElement& r) const
	{
		return l.second > r.second;
	}
};
#endif

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
		const Neighbors& neighbors, DistanceCallback callback)
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

#ifdef TAPKEE_USE_PRIORITY_QUEUE
		reservable_priority_queue<HeapElement,HeapElementComparator> heap(N);
#else
		fibonacci_heap heap(N);
#endif

#pragma omp for nowait
		for (k=0; k<N; k++)
		{
			// fill s and f with false, fill shortest_D with infinity
			for (IndexType j=0; j<N; j++)
			{
				shortest_distances(k,j) = std::numeric_limits<DenseMatrix::Scalar>::max();
				s[j] = false;
				f[j] = false;
			}
			// set distance from k to k as zero
			shortest_distances(k,k) = 0.0;

			// insert kth object to heap with zero distance and set f[k] true
#ifdef TAPKEE_USE_PRIORITY_QUEUE
			HeapElement heap_element_of_self(k,0.0);
			heap.push(heap_element_of_self);
#else
			heap.insert(k,0.0);
#endif
			f[k] = true;

			// while heap is not empty
			while (!heap.empty())
			{
				// extract min and set (s)olution state as true and (f)rontier as false
#ifdef TAPKEE_USE_PRIORITY_QUEUE
				int min_item = heap.top().first;
				ScalarType min_item_d = heap.top().second;
				heap.pop();
				if (min_item_d > shortest_distances(k,min_item))
					continue;
#else
				ScalarType tmp;
				int min_item = heap.extract_min(tmp);
#endif

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
						ScalarType dist = shortest_distances(k,min_item) + callback.distance(begin[min_item],begin[w]);
						// if distance can be relaxed
						if (dist < shortest_distances(k,w))
						{
							// relax distance
							shortest_distances(k,w) = dist;
#ifdef TAPKEE_USE_PRIORITY_QUEUE
							HeapElement relaxed_heap_element(w,dist);
							heap.push(relaxed_heap_element);
							f[w] = true;
#else
							// if w is in (f)rontier
							if (f[w])
							{
								// decrease distance in heap
								heap.decrease_key(w, dist);
							}
							else
							{
								// insert w to heap and set (f)rontier as true
								heap.insert(w, dist);
								f[w] = true;
							}
#endif
						}
					}
				}
			}
			heap.clear();
		}

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
		const Landmarks& landmarks, const Neighbors& neighbors, DistanceCallback callback)
{
	timed_context context("Distances shortest path relaxing");
	const IndexType n_neighbors = neighbors[0].size();
	const IndexType N = end-begin;
	const IndexType N_landmarks = landmarks.size();

	DenseMatrix shortest_distances(landmarks.size(),N);

#pragma omp parallel shared(shortest_distances,begin,landmarks,neighbors,callback) default(none)
	{
		bool* f = new bool[N];
		bool* s = new bool[N];
		IndexType k;

#ifdef TAPKEE_USE_PRIORITY_QUEUE
		reservable_priority_queue<HeapElement,HeapElementComparator> heap(N);
#else
		fibonacci_heap heap(N);
#endif

#pragma omp for nowait
		for (k=0; k<N_landmarks; k++)
		{
			// fill s and f with false, fill shortest_D with infinity
			for (IndexType j=0; j<N; j++)
			{
				shortest_distances(k,j) = std::numeric_limits<DenseMatrix::Scalar>::max();
				s[j] = false;
				f[j] = false;
			}
			// set distance from k to k as zero
			shortest_distances(k,landmarks[k]) = 0.0;

			// insert kth object to heap with zero distance and set f[k] true
#ifdef TAPKEE_USE_PRIORITY_QUEUE
			HeapElement heap_element_of_self(landmarks[k],0.0);
			heap.push(heap_element_of_self);
#else
			heap.insert(landmarks[k],0.0);
#endif
			f[k] = true;

			// while heap is not empty
			while (!heap.empty())
			{
				// extract min and set (s)olution state as true and (f)rontier as false
#ifdef TAPKEE_USE_PRIORITY_QUEUE
				int min_item = heap.top().first;
				ScalarType min_item_d = heap.top().second;
				heap.pop();
				if (min_item_d > shortest_distances(k,min_item))
					continue;
#else
				ScalarType tmp;
				int min_item = heap.extract_min(tmp);
#endif

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
						ScalarType dist = shortest_distances(k,min_item) + callback.distance(begin[min_item],begin[w]);
						// if distance can be relaxed
						if (dist < shortest_distances(k,w))
						{
							// relax distance
							shortest_distances(k,w) = dist;
#ifdef TAPKEE_USE_PRIORITY_QUEUE
							HeapElement relaxed_heap_element(w,dist);
							heap.push(relaxed_heap_element);
							f[w] = true;
#else
							// if w is in (f)rontier
							if (f[w])
							{
								// decrease distance in heap
								heap.decrease_key(w, dist);
							}
							else
							{
								// insert w to heap and set (f)rontier as true
								heap.insert(w, dist);
								f[w] = true;
							}
#endif
						}
					}
				}
			}
			heap.clear();
		}

		delete[] s;
		delete[] f;
	}
	return shortest_distances;
}

}
}

#endif
