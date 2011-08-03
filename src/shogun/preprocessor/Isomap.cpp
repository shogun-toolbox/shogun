/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/preprocessor/Isomap.h>
#ifdef HAVE_LAPACK
#include <shogun/lib/common.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CCustomDistance* CIsomap::isomap_distance(CDistance* distance)
{
	int32_t N,k,i,j;
	float64_t tmp;
	SGMatrix<float64_t> D_matrix=distance->get_distance_matrix();
	N=D_matrix.num_cols;
	ASSERT(m_k<N);
	CFibonacciHeap* heap = new CFibonacciHeap(N);

	// cut by k-nearest neighbors
	int32_t* edges_idx_matrix = SG_MALLOC(int32_t, N*m_k);
	float64_t* edges_matrix = SG_MALLOC(float64_t, N*m_k);
			
	// query neighbors and edges to neighbors
	heap->clear();
	for (i=0; i<N; i++)
	{
		// insert distances to heap
		for (j=0; j<N; j++)
			heap->insert(j,D_matrix.matrix[i*N+j]);

		// extract nearest neighbor: the jth object itself
		heap->extract_min(tmp);

		// extract m_k neighbors and distances
		for (j=0; j<m_k; j++)
		{
			edges_idx_matrix[i*m_k+j] = heap->extract_min(tmp);
			edges_matrix[i*m_k+j] = tmp;
		}
		// clear heap
		heap->clear();
	}
	// cleanup
	D_matrix.destroy_matrix();

	// Dijkstra with Fibonacci Heap (not very efficient yet)
	// allocate (s)olution
	bool* s = SG_MALLOC(bool,N);
	// allocate (f)rontier
	bool* f = SG_MALLOC(bool,N);
	// temporary float to store distance
	float64_t dist;
	// temporary ints to represent nodes
	int32_t min_item, w;
	// init matrix to store shortest distances
	float64_t* shortest_D = SG_MALLOC(float64_t,N*N);
	// clear heap
	heap->clear();

	// for each vertex k
	for (k=0; k<N; k++)
	{
		// fill s and f with false, fill shortest_D with infinity
		for (j=0; j<N; j++)
		{
			shortest_D[k*N+j] = CMath::ALMOST_INFTY;
			s[j] = false;
			f[j] = false;
		}
		// set distance from k to k as zero
		shortest_D[k*N+k] = 0.0;

		// insert kth object to heap with zero distance and set f[k] true
		heap->insert(k,0.0);
		f[k] = true;

		// while heap is not empty
		while (heap->get_num_nodes()>0)
		{
			// extract min and set (s)olution state as true and (f)rontier as false
			min_item = heap->extract_min(tmp);
			s[min_item] = true;
			f[min_item] = false;
			
			// for-each edge (min_item->w)
			for (i=0; i<m_k; i++)
			{
				// get w idx
				w = edges_idx_matrix[min_item*m_k+i];
				// if w is not in solution yet
				if (s[w] == false)
				{
					// get distance from k to i through min_item
					dist = shortest_D[k*N+min_item] + edges_matrix[min_item*m_k+i];
					// if distance can be relaxed
					if (dist < shortest_D[k*N+w])
					{
						// relax distance
						shortest_D[k*N+w] = dist;
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
	// cleanup
	delete heap;
	SG_FREE(edges_matrix);
	SG_FREE(s);
	SG_FREE(f);

	CCustomDistance* geodesic_distance = new CCustomDistance(shortest_D,N,N);

	// should be removed if custom distance doesn't copy the matrix
	SG_FREE(shortest_D);

	return geodesic_distance;
}

#endif /* HAVE_LAPACK */
