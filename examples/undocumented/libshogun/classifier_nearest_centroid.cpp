#include <shogun/features/Labels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/classifier/NearestCentroid.h>

using namespace shogun;

int main(){
	index_t num_vec=7;
	index_t num_feat=2;
	index_t num_class=2;

	// create some data
	SGMatrix<float64_t> matrix(num_feat, num_vec);
	Math::range_fill_vector(matrix.matrix, num_feat*num_vec);

	// Create features ; shogun will now own the matrix created
	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(matrix);

	Math::display_matrix(matrix.matrix,num_feat,num_vec);

	//Create labels
	Labels* labels=new Labels(num_vec);
	for (index_t i=0; i<num_vec; ++i)
		labels->set_label(i, i%num_class);

	//Create Euclidean Distance
	EuclideanDistance* distance = new EuclideanDistance(features,features);

	//Create Nearest Centroid
	CNearestCentroid* nearest_centroid = new CNearestCentroid(distance, labels);
	nearest_centroid->train();

//	classify on training examples
	Labels* output=nearest_centroid->apply();
	Math::display_vector(output->get_labels().vector, output->get_num_labels(),
			"batch output");


	// free up memory

	return 0;
}