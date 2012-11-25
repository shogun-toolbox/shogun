#ifndef TAPKEE_ISOMAP_H_
#define TAPKEE_ISOMAP_H_

#include "../utils/fibonacci_heap.hpp"
#include <limits>
using std::numeric_limits;

//! Computes shortest distances (so-called geodesic distances)
//! using Dijkstra algorithm.
//!
//! @param begin begin data iterator
//! @param end end data iterator
//! @param neighbors neighbors of each vector
//! @param callback distance callback
//!
template <class RandomAccessIterator, class DistanceCallback>
DenseSymmetricMatrix compute_shortest_distances_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
		const Neighbors& neighbors, DistanceCallback callback)
{
	timed_context context("Distances shortest path relaxing");
	const unsigned int n_neighbors = neighbors[0].size();
	const unsigned int N = (end-begin);
	FibonacciHeap* heap = new FibonacciHeap(N);

	bool* s = new bool[N];
	bool* f = new bool[N];

	DenseSymmetricMatrix shortest_distances(N,N);
	
	for (unsigned int k=0; k<N; k++)
	{
		// fill s and f with false, fill shortest_D with infinity
		for (unsigned int j=0; j<N; j++)
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
			DefaultScalarType tmp;
			int min_item = heap->extract_min(tmp);
			s[min_item] = true;
			f[min_item] = false;

			// for-each edge (min_item->w)
			for (unsigned int i=0; i<n_neighbors; i++)
			{
				// get w idx
				int w = neighbors[min_item][i];
				// if w is not in solution yet
				if (s[w] == false)
				{
					// get distance from k to i through min_item
					DefaultScalarType dist = shortest_distances(k,min_item) + callback(begin[min_item],begin[i]);
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
		// clear heap to re-use
		heap->clear();
	}
	delete heap;
	delete[] s;
	delete[] f;
	return shortest_distances;
}

//! Computes shortest distances (so-called geodesic distances)
//! using Dijkstra algorithm.
//!
//! @param begin begin data iterator
//! @param end end data iterator
//! @param neighbors neighbors of each vector
//! @param callback distance callback
//!
template <class RandomAccessIterator, class DistanceCallback>
DenseSymmetricMatrix compute_shortest_distances_matrix(RandomAccessIterator begin, RandomAccessIterator, 
		const Landmarks& landmarks, const Neighbors& neighbors, DistanceCallback callback)
{
	timed_context context("Distances shortest path relaxing");
	const unsigned int n_neighbors = neighbors[0].size();
	const unsigned int N = (landmarks.size());
	FibonacciHeap* heap = new FibonacciHeap(N);

	bool* s = new bool[N];
	bool* f = new bool[N];

	DenseSymmetricMatrix shortest_distances(N,N);
	
	for (unsigned int k=0; k<N; k++)
	{
		// fill s and f with false, fill shortest_D with infinity
		for (unsigned int j=0; j<N; j++)
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
			DefaultScalarType tmp;
			int min_item = heap->extract_min(tmp);
			s[min_item] = true;
			f[min_item] = false;

			// for-each edge (min_item->w)
			for (unsigned int i=0; i<n_neighbors; i++)
			{
				// get w idx
				int w = neighbors[min_item][i];
				// if w is not in solution yet
				if (s[w] == false)
				{
					// get distance from k to i through min_item
					DefaultScalarType dist = shortest_distances(k,min_item) + callback(begin[landmarks[min_item]],begin[landmarks[i]]);
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
		// clear heap to re-use
		heap->clear();
	}
	delete heap;
	delete[] s;
	delete[] f;
	return shortest_distances;
}


#endif
