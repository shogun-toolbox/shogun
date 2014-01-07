/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <base/init.h>
#include <evaluation/CrossValidation.h>
#include <evaluation/ContingencyTableEvaluation.h>
#include <evaluation/StratifiedCrossValidationSplitting.h>
#include <modelselection/GridSearchModelSelection.h>
#include <modelselection/ModelSelectionParameters.h>
#include <modelselection/ParameterCombination.h>
#include <labels/MulticlassLabels.h>
#include <features/DenseFeatures.h>
#include <clustering/KMeans.h>
#include <distance/EuclideanDistance.h>
#include <distance/MinkowskiMetric.h>


using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}


int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	int32_t num_clusters=4;
	int32_t num_features=11;
	int32_t dim_features=3;
	int32_t num_vectors_per_cluster=5;
	float64_t cluster_std_dev=2.0;

	/* build random cluster centers */
	SGMatrix<float64_t> cluster_centers(dim_features, num_clusters);
	SGVector<float64_t>::random_vector(cluster_centers.matrix, dim_features*num_clusters,
			-10.0, 10.0);
	SGMatrix<float64_t>::display_matrix(cluster_centers.matrix, cluster_centers.num_rows,
			cluster_centers.num_cols, "cluster centers");

	/* create data around clusters */
	SGMatrix<float64_t> data(dim_features, num_clusters*num_vectors_per_cluster);
	for (index_t i=0; i<num_clusters; ++i)
	{
		for (index_t j=0; j<dim_features; ++j)
		{
			for (index_t k=0; k<num_vectors_per_cluster; ++k)
			{
				index_t idx=i*dim_features*num_vectors_per_cluster;
				idx+=j;
				idx+=k*dim_features;
				float64_t entry=cluster_centers.matrix[i*dim_features+j];
				data.matrix[idx]=CMath::normal_random(entry, cluster_std_dev);
			}
		}
	}

	/* create features, SG_REF to avoid deletion */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t> ();
	features->set_feature_matrix(data);
	SG_REF(features);

	/* create labels for cluster centers */
	CMulticlassLabels* labels=new CMulticlassLabels(num_features);
	for (index_t i=0; i<num_features; ++i)
		labels->set_label(i, i%2==0 ? 0 : 1);

	/* create distance */
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);

	/* create distance machine */
	CKMeans* clustering=new CKMeans(num_clusters, distance);
	clustering->train(features);

	/* build clusters */
	CMulticlassLabels* result=CLabelsFactory::to_multiclass(clustering->apply());
	for (index_t i=0; i<result->get_num_labels(); ++i)
		SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));

	/* print cluster centers */
	CDenseFeatures<float64_t>* centers=
			(CDenseFeatures<float64_t>*)distance->get_lhs();

	SGMatrix<float64_t> centers_matrix=centers->get_feature_matrix();

	SGMatrix<float64_t>::display_matrix(centers_matrix.matrix, centers_matrix.num_rows,
			centers_matrix.num_cols, "learned centers");

	SGMatrix<float64_t>::display_matrix(cluster_centers.matrix, cluster_centers.num_rows,
			cluster_centers.num_cols, "real centers");

	/* clean up */
	SG_UNREF(result);
	SG_UNREF(centers);
	SG_UNREF(clustering);
	SG_UNREF(labels);
	SG_UNREF(features);

	exit_shogun();

	return 0;
}

