#include <shogun/features/Labels.h>
#include <shogun/io/AsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/classifier/NearestCentroid.h>
#include <shogun/base/init.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(){
	init_shogun(&print_message);
	index_t num_vec=4;
	index_t num_feat=2;
	index_t num_class=4;

	// create some data
	SGMatrix<float64_t> matrix(num_feat, num_vec);
	CMath::range_fill_vector(matrix.matrix, num_feat*num_vec);
	// create vectors
	// shogun will now own the matrix created
	CSimpleFeatures<float64_t>* features=new CSimpleFeatures<float64_t>(matrix);

	CMath::display_matrix(matrix.matrix,num_feat,num_vec);
	// create three labels
	CLabels* labels=new CLabels(num_vec);
	for (index_t i=0; i<num_vec; ++i)
		labels->set_label(i, i%num_class);

	// create gaussian kernel with cache 10MB, width 0.5
	CEuclidianDistance* distance = new CEuclidianDistance(features,features);

	// create libsvm with C=10 and train
	CNearestCentroid* nearest_centroid = new CNearestCentroid(distance, labels);
	nearest_centroid->train();
// 
// 	// classify on training examples
	CLabels* output=nearest_centroid->apply();
	CMath::display_vector(output->get_labels().vector, output->get_num_labels(),
			"batch output");
	
	SG_UNREF(output);

	// free up memory
	SG_UNREF(nearest_centroid);

	exit_shogun();
	return 0;
	
}