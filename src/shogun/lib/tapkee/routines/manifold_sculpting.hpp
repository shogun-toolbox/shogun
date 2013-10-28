/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Vladyslav Gorbatiuk
 */

#ifndef TAPKEE_MANIFOLD_SCULPTING_H_
#define TAPKEE_MANIFOLD_SCULPTING_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
#include <shogun/lib/tapkee/utils/sparse.hpp>
/* End of Tapkee includes */

#include <math.h>
#include <ctime>
#include <cfloat>
#include <deque>
#include <set>

namespace tapkee
{
namespace tapkee_internal
{

namespace
{
	const ScalarType max_number_of_iterations_without_improvement = 20;
	const ScalarType multiplier_treshold = 0.01;
	const ScalarType weight_for_adjusted_point = 10.0;
	const ScalarType learning_rate_grow_factor = 1.1;
	const ScalarType learning_rate_shrink_factor = 0.9;
}

/** @brief Data needed to compute error function
 */
struct DataForErrorFunc
{
	/** contains distances between point and its neighbors */
	const SparseMatrix& distance_matrix;
	/** sparse matrix that contains
	 * original angles between the point, its neighbor and
	 * the most collinear neighbor of the neighbor. If
	 * point's index is P, its neighbor's index is N1 and
	 * the index of neighbor's neighbor is N2, then the
	 * angle between them should be stored at index (P, N2)
	 */
	const SparseMatrix& angles_matrix;
	/** a vector of vectors, where I'th
	 * vector contains indices of neighbors for I'th point
	 */
	const Neighbors& distance_neighbors;
	/** a vector of vectors,
	 * where the vector at index I contains indices of
	 * neighbor's neighbor of the I'th point (so that
	 * we know, where to search for the angle value)
	 */
	const Neighbors& angle_neighbors;
	/** a set of indices of points, that have been
	 * already adjusted
	 */
	const std::set<IndexType>& adjusted_points;
	/** initial average distance between neighbors */
	const ScalarType average_distance;
};

template<class RandomAccessIterator, class DistanceCallback>
SparseMatrix neighbors_distances_matrix(RandomAccessIterator begin, RandomAccessIterator end,
                                        const Neighbors& neighbors, DistanceCallback callback,
                                        ScalarType& average_distance)
{
	const IndexType k = neighbors[0].size();
	const IndexType n = neighbors.size();
	if ((end-begin)!=n)
		throw std::runtime_error("Wrong size");
	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*n);
	average_distance = 0;
	ScalarType current_distance;

	for (IndexType i = 0; i < n; ++i)
	{
		const LocalNeighbors& current_neighbors = neighbors[i];
		for (IndexType j = 0; j < k; ++j)
		{
			current_distance = callback.distance(begin[i], begin[current_neighbors[j]]);
			average_distance += current_distance;
			SparseTriplet triplet(i, current_neighbors[j], current_distance);
			sparse_triplets.push_back(triplet);
		}
	}
	average_distance /= (k*n);
	return sparse_matrix_from_triplets(sparse_triplets, n, n);
}

SparseMatrixNeighborsPair angles_matrix_and_neighbors(const Neighbors& neighbors,
                                                      const DenseMatrix& data)
{
	const IndexType k = neighbors[0].size();
	const IndexType n_vectors = data.cols();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k * n_vectors);
	/* I tried to find better naming, but... */
	Neighbors most_collinear_neighbors_of_neighbors;
	most_collinear_neighbors_of_neighbors.reserve(n_vectors);

	for (IndexType i = 0; i < n_vectors; ++i)
	{
		const LocalNeighbors& current_neighbors = neighbors[i];
		LocalNeighbors most_collinear_current_neighbors;
		most_collinear_current_neighbors.reserve(k);

		for (IndexType j = 0; j < k; ++j)
		{
			const LocalNeighbors& neighbors_of_neighbor = neighbors[current_neighbors[j]];
			/* The closer the cos value to -1.0 - the closer the angle to 180.0 */
			ScalarType min_cos_value = 1.0, current_cos_value;
			/* This value will be updated during the seach for most collinear neighbor */
			most_collinear_current_neighbors.push_back(0);

			for (IndexType l = 0; l < k; ++l)
			{
				DenseVector neighbor_to_point = data.col(i) - data.col(current_neighbors[j]);
				DenseVector neighbor_to_its_neighbor = data.col(neighbors_of_neighbor[l])
														- data.col(current_neighbors[j]);
				current_cos_value = neighbor_to_point.dot(neighbor_to_its_neighbor) /
									(neighbor_to_point.norm() *
									 neighbor_to_its_neighbor.norm());
				if (current_cos_value < min_cos_value)
				{
					most_collinear_current_neighbors[j] = neighbors_of_neighbor[l];
					min_cos_value = current_cos_value;
				}
			}

			SparseTriplet triplet(i, most_collinear_current_neighbors[j], min_cos_value);
			sparse_triplets.push_back(triplet);
		}
		most_collinear_neighbors_of_neighbors.push_back(most_collinear_current_neighbors);
	}
	return SparseMatrixNeighborsPair
		(sparse_matrix_from_triplets(sparse_triplets, n_vectors, n_vectors),
		 most_collinear_neighbors_of_neighbors);
}

ScalarType average_neighbor_distance(const DenseMatrix& data, const Neighbors& neighbors)
{
	IndexType k = neighbors[0].size();
	ScalarType average_distance = 0;

	for (IndexType i = 0; i < data.cols(); ++i)
	{
		for (IndexType j = 0; j < k; ++j)
		{
			average_distance += (data.col(i) - data.col(neighbors[i][j])).norm();
		}
	}
	return average_distance / (k * data.cols());
}

ScalarType compute_error_for_point(const IndexType index, const DenseMatrix& data,
                                   const DataForErrorFunc& error_func_data)
{
	IndexType k = error_func_data.distance_neighbors[0].size();
	ScalarType error_value = 0;
	for (IndexType i = 0; i < k; ++i)
	{
		/* Find new angle */
		IndexType neighbor = error_func_data.distance_neighbors[index][i];
		IndexType neighbor_of_neighbor = error_func_data.angle_neighbors[index][i];
		/* TODO: Extract into a small function, that will find the angle between 3 points */
		DenseVector neighbor_to_point = data.col(index) - data.col(neighbor);
		DenseVector neighbor_to_its_neighbor = data.col(neighbor_of_neighbor)
												- data.col(neighbor);
		ScalarType current_cos_value = neighbor_to_point.dot(neighbor_to_its_neighbor) /
									(neighbor_to_point.norm() *
									 neighbor_to_its_neighbor.norm());
		/* Find new distance */
		ScalarType current_distance = (data.col(index) - data.col(neighbor)).norm();
		/* Compute one component of error function's value*/
		ScalarType diff_cos =
			current_cos_value - error_func_data.angles_matrix.coeff(index, neighbor_of_neighbor);
		if (diff_cos < 0)
			diff_cos = 0;
		ScalarType diff_distance =
			current_distance - error_func_data.distance_matrix.coeff(index, neighbor);
		diff_distance /= error_func_data.average_distance;
		/* Weight for adjusted point should be bigger than 1, according to the
		 * original algorithm
		 */
		ScalarType weight =
			(error_func_data.adjusted_points.count(neighbor) == 0) ? 1 : weight_for_adjusted_point;
		error_value += weight * (diff_cos * diff_cos + diff_distance * diff_distance);
	}
	return error_value;
}

/** Adjust the data point at given index to restore
 * the original relationships between point and its
 * neighbors. Uses simple hill-climbing technique
 * @param index index of the point to adjust
 * @param data the data matrix - will be changed after
 * the point adjustment (only index column will change)
 * @param target_dimension - you know, what it is:)
 * @param learning_rate some small value, that will be
 * used during hill-climbing to change the points coordinates
 * @param error_func_data a special struct, that contains
 * data, needed for error function calculation - such
 * as initial distances between neighbors, initial
 * angles, etc.
 * @param point_error - will be set to the error function
 * value, calculated for the point
 * @return a number of steps it took to  adjust the
 * point
 */
IndexType adjust_point_at_index(const IndexType index, DenseMatrix& data,
                                const IndexType target_dimension,
                                const ScalarType learning_rate,
                                const DataForErrorFunc& error_func_data,
                                ScalarType& point_error)
{
	IndexType n_steps = 0;
	ScalarType old_error, new_error;
	bool finish = false;
	while (!finish)
	{
		finish = true;
		old_error = compute_error_for_point(index, data, error_func_data);

		for (IndexType i = 0; i < target_dimension; ++i)
		{
			/* Try to change the current coordinate in positive direction */
			data(i, index) += learning_rate;
			new_error = compute_error_for_point(index, data, error_func_data);
			if (new_error >= old_error)
			{
				/* Did not help - switching to negative direction */
				data(i, index) -= 2 * learning_rate;
				new_error = compute_error_for_point(index, data, error_func_data);

			}
			if (new_error >= old_error)
				/* Did not help again - reverting to beginning */
				data(i, index) += learning_rate;
			else
			{
				/* We made some progress (improved an error) */
				old_error = new_error;
				finish = false;
			}
		}
		++n_steps;
	}
	point_error = compute_error_for_point(index, data, error_func_data);
	return n_steps;
}

template <class RandomAccessIterator, class DistanceCallback>
void manifold_sculpting_embed(RandomAccessIterator begin, RandomAccessIterator end,
                              DenseMatrix& data, IndexType target_dimension,
                              const Neighbors& neighbors, DistanceCallback callback,
                              IndexType max_iteration, ScalarType squishing_rate)
{
	/* Step 1: Get initial distances to each neighbor and initial
	 * angles between the point Pi, each neighbor Nij, and the most
	 * collinear neighbor of Nij.
	 */
	ScalarType initial_average_distance;
	SparseMatrix distances_to_neighbors =
		neighbors_distances_matrix(begin, end, neighbors, callback, initial_average_distance);
	SparseMatrixNeighborsPair angles_and_neighbors =
		angles_matrix_and_neighbors(neighbors, data);

	/* Step 2: Optionally preprocess the data using PCA
	 * (skipped for now).
	 */
	ScalarType no_improvement_counter = 0, normal_counter = 0;
	ScalarType current_multiplier = squishing_rate;
	ScalarType learning_rate = initial_average_distance;
	ScalarType best_error = DBL_MAX, current_error, point_error;
	/* Step 3: Do until no improvement is made for some period
	 * (or until max_iteration number is reached):
	 */
	while (((no_improvement_counter++ < max_number_of_iterations_without_improvement)
			|| (current_multiplier >  multiplier_treshold))
			&& (normal_counter++ < max_iteration))
	{
		/* Step 3a: Scale the data in non-preserved dimensions
		 * by a factor of squishing_rate.
		 */
		data.bottomRows(data.rows() - target_dimension) *= squishing_rate;
		while (average_neighbor_distance(data, neighbors) < initial_average_distance)
		{
			data.topRows(target_dimension) /= squishing_rate;
		}
		current_multiplier *= squishing_rate;

		/* Step 3b: Restore the previously computed relationships
		 * (distances to neighbors and angles to ...) by adjusting
		 * data points in first target_dimension dimensions.
		 */
		/* Start adjusting from a random point */
		IndexType start_point_index = std::rand() % data.cols();
		std::deque<IndexType> points_to_adjust;
		points_to_adjust.push_back(start_point_index);
		ScalarType steps_made = 0;
		current_error = 0;
		std::set<IndexType> adjusted_points;

		while (!points_to_adjust.empty())
		{
			IndexType current_point_index = points_to_adjust.front();
			points_to_adjust.pop_front();
			if (adjusted_points.count(current_point_index) == 0)
			{
			DataForErrorFunc error_func_data = {
					distances_to_neighbors,
					angles_and_neighbors.first,
					neighbors,
					angles_and_neighbors.second,
					adjusted_points,
					initial_average_distance
				};
				adjust_point_at_index(current_point_index, data, target_dimension,
									learning_rate, error_func_data, point_error);
				current_error += point_error;
				/* Insert all neighbors into deque */
				std::copy(neighbors[current_point_index].begin(),
				          neighbors[current_point_index].end(),
				          std::back_inserter(points_to_adjust));
				adjusted_points.insert(current_point_index);
			}
		}

		if (steps_made > data.cols())
			learning_rate *= learning_rate_grow_factor;
		else
			learning_rate *= learning_rate_shrink_factor;

		if (current_error < best_error)
		{
			best_error = current_error;
			no_improvement_counter = 0;
		}
	}

	data.conservativeResize(target_dimension, Eigen::NoChange);
	data.transposeInPlace();
}

}
}

#endif
