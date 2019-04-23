/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Soeren Sonnenburg, Heiko Strathmann,
 *          Evgeniy Andreev, Chiyuan Zhang, Thoralf Klein, Sergey Lisitsyn
 */

#include <shogun/base/init.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/distance/MinkowskiMetric.h>


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
				data.matrix[idx]=Math::normal_random(entry, cluster_std_dev);
			}
		}
	}

	auto features=std::make_shared<DenseFeatures<float64_t>>(data);

	/* create labels for cluster centers */
	auto labels=std::make_shared<MulticlassLabels>(num_features);
	for (index_t i=0; i<num_features; ++i)
		labels->set_label(i, i%2==0 ? 0 : 1);

	/* create distance */
	auto distance=std::make_shared<EuclideanDistance>(features, features);

	/* create distance machine */
	auto clustering=std::make_shared<KMeans>(num_clusters, distance);
	clustering->train(features);

	/* build clusters */
	auto result = clustering->apply()->as<MulticlassLabels>();
	for (index_t i=0; i<result->get_num_labels(); ++i)
		SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));

	/* print cluster centers */
	auto centers=distance->get_lhs()->as<DenseFeatures<float64_t>>();

	SGMatrix<float64_t> centers_matrix=centers->get_feature_matrix();

	SGMatrix<float64_t>::display_matrix(centers_matrix.matrix, centers_matrix.num_rows,
			centers_matrix.num_cols, "learned centers");

	SGMatrix<float64_t>::display_matrix(cluster_centers.matrix, cluster_centers.num_rows,
			cluster_centers.num_cols, "real centers");

	/* clean up */

	exit_shogun();

	return 0;
}

