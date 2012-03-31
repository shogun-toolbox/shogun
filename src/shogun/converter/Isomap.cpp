/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/Isomap.h>
#ifdef HAVE_LAPACK
#include <shogun/lib/common.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parallel.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/CoverTree.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/* struct storing thread params
 */
struct DIJKSTRA_THREAD_PARAM
{
	/// fibonacci heap to be used
	CFibonacciHeap* heap;
	/// const matrix storing edges lengths (ith column
	/// contains lengths from ith object)
	const float64_t* edges_matrix;
	/// const matrix storing edges idxs (ith column
	/// contains indexes of end-point vertices)
	const int32_t* edges_idx_matrix;
	/// matrix to store shortest paths
	float64_t* shortest_D;
	/// starting index of loop
	int32_t i_start;
	/// stopping index of loop
	int32_t i_stop;
	/// step for loop
	int32_t i_step;
	/// k param
	int32_t m_k;
	/// (s)olution bool array
	bool* s;
	/// (f)rontier bool array
	bool* f;
};

class ISOMAP_COVERTREE_POINT
{
public:

	ISOMAP_COVERTREE_POINT(int32_t index, const SGMatrix<float64_t>& dmatrix)
	{
		point_index = index;
		distance_matrix = dmatrix;
	}

	inline double distance(const ISOMAP_COVERTREE_POINT& p) const
	{
		return distance_matrix[point_index*distance_matrix.num_rows+p.point_index];
	}

	inline bool operator==(const ISOMAP_COVERTREE_POINT& p) const
	{
		return (p.point_index==point_index);
	}

	int32_t point_index;
	SGMatrix<float64_t> distance_matrix;
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CIsomap::CIsomap() : CMultidimensionalScaling()
{
	m_k = 3;

	init();
}

void CIsomap::init()
{
	m_parameters->add(&m_k, "k", "number of neighbors");
}

CIsomap::~CIsomap()
{
}

void CIsomap::set_k(int32_t k)
{
	ASSERT(k>0);
	m_k = k;
}

int32_t CIsomap::get_k() const
{
	return m_k;
}

const char* CIsomap::get_name() const
{
	return "Isomap";
}

SGMatrix<float64_t> CIsomap::process_distance_matrix(SGMatrix<float64_t> distance_matrix)
{
	return isomap_distance(distance_matrix);
}

SGMatrix<float64_t> CIsomap::isomap_distance(SGMatrix<float64_t> D_matrix)
{
	int32_t N,i;
	N = D_matrix.num_cols;
	if (D_matrix.num_cols!=D_matrix.num_rows)
	{
		D_matrix.destroy_matrix();
		SG_ERROR("Given distance matrix is not square.\n");
	}
	if (m_k>=N)
	{
		D_matrix.destroy_matrix();
		SG_ERROR("K parameter should be less than number of given vectors (k=%d, N=%d)\n", m_k, N);
	}

	// cut by k-nearest neighbors
	int32_t* edges_idx_matrix = SG_MALLOC(int32_t, N*m_k);
	float64_t* edges_matrix = SG_MALLOC(float64_t, N*m_k);

	float64_t max_dist = CMath::max(D_matrix.matrix,N*N);
	CoverTree<ISOMAP_COVERTREE_POINT>* coverTree = new CoverTree<ISOMAP_COVERTREE_POINT>(max_dist);

	for (i=0; i<N; i++)
		coverTree->insert(ISOMAP_COVERTREE_POINT(i,D_matrix));

	for (i=0; i<N; i++)
	{
		ISOMAP_COVERTREE_POINT origin(i,D_matrix);
		std::vector<ISOMAP_COVERTREE_POINT> neighbors =
		   coverTree->kNearestNeighbors(origin,m_k+1);
		for (std::size_t m=1; m<neighbors.size(); m++)
		{
			edges_matrix[i*m_k+m-1] = origin.distance(neighbors[m]);
			edges_idx_matrix[i*m_k+m-1] = neighbors[m].point_index;
		}
	}

	delete coverTree;

#ifdef HAVE_PTHREAD
	int32_t t;
	// Parallel Dijkstra with Fibonacci Heap
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads and thread parameters
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	DIJKSTRA_THREAD_PARAM* parameters = SG_MALLOC(DIJKSTRA_THREAD_PARAM, num_threads);
	// allocate heaps
	CFibonacciHeap** heaps = SG_MALLOC(CFibonacciHeap*, num_threads);
	for (t=0; t<num_threads; t++)
		heaps[t] = new CFibonacciHeap(N);

#else
	int32_t num_threads = 1;
#endif

	// allocate (s)olution
	bool* s = SG_MALLOC(bool,N*num_threads);
	// allocate (f)rontier
	bool* f = SG_MALLOC(bool,N*num_threads);
	// init matrix to store shortest distances
	float64_t* shortest_D = D_matrix.matrix;

#ifdef HAVE_PTHREAD

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for (t=0; t<num_threads; t++)
	{
		parameters[t].i_start = t;
		parameters[t].i_stop = N;
		parameters[t].i_step = num_threads;
		parameters[t].heap = heaps[t];
		parameters[t].edges_matrix = edges_matrix;
		parameters[t].edges_idx_matrix = edges_idx_matrix;
		parameters[t].s = s+t*N;
		parameters[t].f = f+t*N;
		parameters[t].m_k = m_k;
		parameters[t].shortest_D = shortest_D;
		pthread_create(&threads[t], &attr, CIsomap::run_dijkstra_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr);
	for (t=0; t<num_threads; t++)
		delete heaps[t];
	SG_FREE(heaps);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	DIJKSTRA_THREAD_PARAM single_thread_param;
	single_thread_param.i_start = 0;
	single_thread_param.i_stop = N;
	single_thread_param.i_step = 1;
	single_thread_param.m_k = m_k;
	single_thread_param.heap = new CFibonacciHeap(N);
	single_thread_param.edges_matrix = edges_matrix;
	single_thread_param.edges_idx_matrix = edges_idx_matrix;
	single_thread_param.s = s;
	single_thread_param.f = f;
	single_thread_param.shortest_D = shortest_D;
	run_dijkstra_thread((void*)&single_thread_param);
	delete single_thread_param.heap;
#endif
	// cleanup
	SG_FREE(edges_matrix);
	SG_FREE(edges_idx_matrix);
	SG_FREE(s);
	SG_FREE(f);

	return SGMatrix<float64_t>(shortest_D,N,N);
}

void* CIsomap::run_dijkstra_thread(void *p)
{
	// get parameters from p
	DIJKSTRA_THREAD_PARAM* parameters = (DIJKSTRA_THREAD_PARAM*)p;
	CFibonacciHeap* heap = parameters->heap;
	int32_t i_start = parameters->i_start;
	int32_t i_stop = parameters->i_stop;
	int32_t i_step = parameters->i_step;
	bool* s = parameters->s;
	bool* f = parameters->f;
	const float64_t* edges_matrix = parameters->edges_matrix;
	const int32_t* edges_idx_matrix = parameters->edges_idx_matrix;
	float64_t* shortest_D = parameters->shortest_D;
	int32_t m_k = parameters->m_k;
	int32_t k,j,i,min_item,w;
	int32_t N = i_stop;

	// temporary vars
	float64_t dist,tmp;

	// main loop
	for (k=i_start; k<i_stop; k+=i_step)
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
	return NULL;
}

#endif /* HAVE_LAPACK */
