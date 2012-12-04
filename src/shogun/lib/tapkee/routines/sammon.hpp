/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_SAMMON_H_
#define TAPKEE_SAMMON_H_

template <class RandomAccessIterator, class DistanceCallback>
EmbeddingResult sammon_mapping(RandomAccessIterator begin, RandomAccessIterator end,
		DistanceCallback callback,
		unsigned int target_dimension, DefaultScalarType tolerance)
{
	DenseMatrix distance_matrix = compute_distance_matrix(begin,end,callback);
	DefaultScalarType scale = 0.5 / distance_matrix.array().sum();
	distance_matrix.noalias() += DenseMatrix::Identity((end-begin),(end-begin));

	DenseMatrix Y = DenseMatrix::Random(target_dimension, (end-begin));

	return EmbeddingResult(Y,DenseVector());
}

#endif
