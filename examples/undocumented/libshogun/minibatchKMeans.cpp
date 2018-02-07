/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Björn Esser
 */

#include <shogun/base/init.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	int32_t dim_features=2;

	/* create data around clusters */
	SGMatrix<float64_t> data(dim_features, 4);
	data(0,0)=0;
	data(0,1)=0;
	data(0,2)=2;
	data(0,3)=2;
	data(1,0)=0;
	data(1,1)=1000;
	data(1,2)=1000;
	data(1,3)=0;
	data.display_matrix(data.matrix, 2, 4, "rectangle_coordinates");


	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t> (data);
	SG_REF(features);

	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKMeans* clustering=new CKMeans(2, distance, true);

	clustering->train(features);
	CMulticlassLabels* result=CLabelsFactory::to_multiclass(clustering->apply());

	for (index_t i=0; i<result->get_num_labels(); ++i)
		SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));

	CDenseFeatures<float64_t>* centers=distance->get_lhs()->as<CDenseFeatures<float64_t>>();
	SGMatrix<float64_t> centers_matrix=centers->get_feature_matrix();
	centers_matrix.display_matrix(centers_matrix.matrix,
			centers_matrix.num_rows, centers_matrix.num_cols, "learnt centers using Lloyd's KMeans");

	SG_UNREF(centers);
	SG_UNREF(result);

	clustering->set_train_method(KMM_MINI_BATCH);
	clustering->set_mbKMeans_params(2,10);
	clustering->train(features);
	result=CLabelsFactory::to_multiclass(clustering->apply());

	for (index_t i=0; i<result->get_num_labels(); ++i)
		SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));

	centers=(CDenseFeatures<float64_t>*)distance->get_lhs();
	centers_matrix=centers->get_feature_matrix();
	centers_matrix.display_matrix(centers_matrix.matrix, centers_matrix.num_rows,
			centers_matrix.num_cols, "learnt centers using mini-batch KMeans");

	SG_UNREF(centers);
	SG_UNREF(result);
	SG_UNREF(clustering);
	SG_UNREF(features);

	exit_shogun();

	return 0;
}

