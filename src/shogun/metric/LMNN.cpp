/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2013 Fernando J. Iglesias Garcia
 */

#ifdef HAVE_EIGEN3

#include <shogun/metric/LMNN.h>
#include <Eigen/Dense>

using namespace shogun;
using namespace Eigen;

CLMNN::CLMNN()
{
	init();
}

CLMNN::CLMNN(CFeatures* features, CMulticlassLabels* labels)
{
	init();

	m_features = features;
	m_labels   = labels;

	SG_REF(m_features)
	SG_REF(m_labels)
}

CLMNN::~CLMNN()
{
	SG_UNREF(m_features)
	SG_UNREF(m_labels)
}

const char* CLMNN::get_name() const
{
	return "LMNN";
}

//TODO
void CLMNN::train(SGMatrix<float64_t> init_transform)
{
}

SGMatrix<float64_t> CLMNN::get_linear_transform() const
{
	return m_linear_transform;
}

CCustomMahalanobisDistance* CLMNN::get_distance() const
{
	// Compute Mahalanobis distance matrix M = L^T*L

	// Put the linear transform L in Eigen to perform the matrix multiplication
	// L is not copied to another region of memory
	Map<const MatrixXd> map_linear_transform(m_linear_transform.matrix, m_linear_transform.num_rows,
			m_linear_transform.num_cols);
	// TODO exploit that M is symmetric
	MatrixXd M = map_linear_transform.transpose()*map_linear_transform;
	// TODO avoid copying
	SGMatrix<float64_t> mahalanobis_matrix(M.rows(), M.cols());
	for (index_t i = 0; i < M.rows(); i++)
		for (index_t j = 0; j < M.cols(); j++)
			mahalanobis_matrix(i,j) = M(i,j);

	// Create custom Mahalanobis distance with matrix M associated with the training features

	CCustomMahalanobisDistance* distance =
			new CCustomMahalanobisDistance(m_features, m_features, mahalanobis_matrix);
	SG_REF(distance)

	return distance;
}

void CLMNN::init()
{
	SG_ADD(&m_linear_transform, "m_linear_transform", "Linear transform in matrix form", MS_NOT_AVAILABLE)
	SG_ADD((CSGObject**) &m_features, "m_features", "Training features", MS_NOT_AVAILABLE)
	SG_ADD((CSGObject**) &m_labels, "m_labels", "Training labels", MS_NOT_AVAILABLE)

	m_features = NULL;
	m_labels   = NULL;
}

#endif /* HAVE_EIGEN3 */
