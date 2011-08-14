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
#include <shogun/base/Parallel.h>
#include <shogun/lib/Signal.h>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/* struct storing thread params
 */
struct D_THREAD_PARAM
{
	// heap to be used
	CFibonacciHeap* heap;
	// const matrix storing edges lengths
	const float64_t* edges_matrix;
	// const matrix storing edges idxs
	const int32_t* edges_idx_matrix;
	// matrix to store shortest paths
	float64_t* shortest_D;
	// idx of threads start
	int32_t idx_start;
	// idx of thread stop
	int32_t idx_stop;
	// idx of thread step
	int32_t idx_step;
	// k param
	int32_t m_k;
	// (s)olution bool array
	bool* s;
	// (f)rontier bool array
	bool* f;
};
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CSimpleFeatures<float64_t>* CIsomap::apply_to_distance(CDistance* distance)
{
	CDistance* geodesic_distance = isomap_distance(distance);

	CSimpleFeatures<float64_t>* new_features = CMultidimensionalScaling::apply_to_distance(geodesic_distance);

	delete geodesic_distance;

	return new_features;
}


SGMatrix<float64_t> CIsomap::apply_to_feature_matrix(CFeatures* features)
{
	CSimpleFeatures<float64_t>* simple_features =
		(CSimpleFeatures<float64_t>*) features;
	CDistance* euclidian_distance = new CEuclidianDistance();
	euclidian_distance->init(simple_features,simple_features);
	CDistance* geodesic_distance = isomap_distance(euclidian_distance);

	SGMatrix<float64_t> new_features;

	if (m_landmark) 
		new_features = CMultidimensionalScaling::landmark_embedding(geodesic_distance);
	else
		new_features = CMultidimensionalScaling::classic_embedding(geodesic_distance);

	delete geodesic_distance;
	delete euclidian_distance;

	simple_features->set_feature_matrix(new_features);

	return new_features;
}
	
SGVector<float64_t> CIsomap::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}
	
CCustomDistance* CIsomap::isomap_distance(CDistance* distance)
{
	int32_t N,t,i,j;
	float64_t tmp;
	SGMatrix<float64_t> D_matrix=distance->get_distance_matrix();
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
			
	// query neighbors and edges to neighbors
	CFibonacciHeap* heap = new CFibonacciHeap(N);
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
	delete heap;
	D_matrix.destroy_matrix();

#ifndef WIN32

	// Parallel Dijkstra with Fibonacci Heap 
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads and thread parameters
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	D_THREAD_PARAM* parameters = SG_MALLOC(D_THREAD_PARAM, num_threads);
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
	float64_t* shortest_D = SG_MALLOC(float64_t,N*N);

#ifndef WIN32

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for (t=0; t<num_threads; t++)
	{
		parameters[t].idx_start = t;	
		parameters[t].idx_stop = N;	
		parameters[t].idx_step = num_threads;	
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
	D_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_stop = N;
	single_thread_param.idx_step = 1;
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

	CCustomDistance* geodesic_distance = new CCustomDistance(shortest_D,N,N);

	// should be removed if custom distance doesn't copy the matrix
	SG_FREE(shortest_D);

	return geodesic_distance;
}

void* CIsomap::run_dijkstra_thread(void *p)
{	
	D_THREAD_PARAM* parameters = (D_THREAD_PARAM*)p;
	CFibonacciHeap* heap = parameters->heap;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_stop = parameters->idx_stop;
	int32_t idx_step = parameters->idx_step;
	bool* s = parameters->s;
	bool* f = parameters->f;
	const float64_t* edges_matrix = parameters->edges_matrix;
	const int32_t* edges_idx_matrix = parameters->edges_idx_matrix;
	float64_t* shortest_D = parameters->shortest_D;
	int32_t m_k = parameters->m_k;
	int32_t k,j,i,min_item,w;
	int32_t N = idx_stop;
	float64_t dist,tmp;

	for (k=idx_start; k<idx_stop; k+=idx_step)
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
