/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_MDS_H_
#define TAPKEE_MDS_H_

#include <algorithm>

using std::random_shuffle;
using std::fill;

namespace tapkee
{
namespace tapkee_internal
{

template <class RandomAccessIterator>
Landmarks select_landmarks_random(RandomAccessIterator begin, RandomAccessIterator end, ScalarType ratio)
{
	Landmarks landmarks;
	landmarks.reserve(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
		landmarks.push_back(iter-begin);
	random_shuffle(landmarks.begin(),landmarks.end());
	landmarks.erase(landmarks.begin() + static_cast<IndexType>(landmarks.size()*ratio),landmarks.end());
	return landmarks;
}

template <class RandomAccessIterator, class PairwiseCallback>
DenseSymmetricMatrix compute_distance_matrix(RandomAccessIterator begin, Landmarks landmarks, 
                                             PairwiseCallback callback)
{
	timed_context context("Multidimensional scaling distance matrix computation");

	DenseMatrix distance_matrix(landmarks.size(),landmarks.size());

	for (Landmarks::const_iterator i_iter=landmarks.begin(); i_iter!=landmarks.end(); ++i_iter)
	{
		for (Landmarks::const_iterator j_iter=i_iter; j_iter!=landmarks.end(); ++j_iter)
		{
			ScalarType d = callback(*(begin+*i_iter),*(begin+*j_iter));
			d *= d;
			distance_matrix(i_iter-landmarks.begin(),j_iter-landmarks.begin()) = d;
			distance_matrix(j_iter-landmarks.begin(),i_iter-landmarks.begin()) = d;
		}
	}
	return distance_matrix;
};

template <class RandomAccessIterator, class PairwiseCallback>
EmbeddingResult triangulate(RandomAccessIterator begin, RandomAccessIterator end, PairwiseCallback distance_callback,
                            const Landmarks& landmarks, const DenseVector& landmark_distances_squared, 
                            EmbeddingResult& landmarks_embedding, IndexType target_dimension)
{
	timed_context context("Landmark triangulation");
	
	bool* to_process = new bool[end-begin];
	fill(to_process,to_process+(end-begin),true);
	
	DenseMatrix embedding((end-begin),target_dimension);

	for (Landmarks::const_iterator iter=landmarks.begin(); 
			iter!=landmarks.end(); ++iter)
	{
		to_process[*iter] = false;
		embedding.row(*iter).noalias() = landmarks_embedding.first.row(iter-landmarks.begin());
	}

	for (IndexType i=0; i<target_dimension; ++i)
		landmarks_embedding.first.col(i).array() /= landmarks_embedding.second(i);

	RandomAccessIterator iter;
	DenseVector distances_to_landmarks;

//#pragma omp parallel private(distances_to_landmarks)
	{
	distances_to_landmarks = DenseVector(landmarks.size());
//#pragma omp for private(iter) schedule(static)
	for (iter=begin; iter<end; ++iter)
	{
		if (!to_process[iter-begin])
			continue;

		for (IndexType i=0; i<distances_to_landmarks.size(); ++i)
		{
			ScalarType d = distance_callback(*iter,begin[landmarks[i]]);
			distances_to_landmarks(i) = d*d;
		}
		//distances_to_landmarks.array().square();

		distances_to_landmarks -= landmark_distances_squared;
		embedding.row(iter-begin).noalias() = -0.5*landmarks_embedding.first.transpose()*distances_to_landmarks;
	}
	}

	delete[] to_process;

	return EmbeddingResult(embedding,DenseVector());
}

template <class RandomAccessIterator, class PairwiseCallback>
DenseSymmetricMatrix compute_distance_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
                                             PairwiseCallback callback)
{
	timed_context context("Multidimensional scaling distance matrix computation");

	DenseMatrix distance_matrix(end-begin,end-begin);

	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=i_iter; j_iter!=end; ++j_iter)
		{
			ScalarType d = callback(*i_iter,*j_iter);
			d *= d;
			distance_matrix(i_iter-begin,j_iter-begin) = d;
			distance_matrix(j_iter-begin,i_iter-begin) = d;
		}
	}
	return distance_matrix;
};

}
}

#endif
