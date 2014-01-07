#include <features/Labels.h>
#include <features/DenseFeatures.h>
#include <distance/EuclideanDistance.h>
#include <classifier/NearestCentroid.h>
#include <base/init.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(){
	init_shogun(&print_message);
	index_t num_vec=7;
	index_t num_feat=2;
	index_t num_class=2;

	// create some data
	SGMatrix<float64_t> matrix(num_feat, num_vec);
	CMath::range_fill_vector(matrix.matrix, num_feat*num_vec);

	// Create features ; shogun will now own the matrix created
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(matrix);

	CMath::display_matrix(matrix.matrix,num_feat,num_vec);

	//Create labels
	CLabels* labels=new CLabels(num_vec);
	for (index_t i=0; i<num_vec; ++i)
		labels->set_label(i, i%num_class);

	//Create Euclidean Distance
	CEuclideanDistance* distance = new CEuclideanDistance(features,features);

	//Create Nearest Centroid
	CNearestCentroid* nearest_centroid = new CNearestCentroid(distance, labels);
	nearest_centroid->train();

//	classify on training examples
	CLabels* output=nearest_centroid->apply();
	CMath::display_vector(output->get_labels().vector, output->get_num_labels(),
			"batch output");

	SG_UNREF(output);

	// free up memory
	SG_UNREF(nearest_centroid);

	exit_shogun();
	return 0;

}