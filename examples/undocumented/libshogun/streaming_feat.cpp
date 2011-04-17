#include <shogun/features/StreamingFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

#include <stdlib.h>
#include <stdio.h>

using namespace shogun;

int main()
{

	init_shogun();

	FILE* infile=fopen("sample.dat", "r");
	CStreamingFeatures* features = new CStreamingFeatures(infile, 100);

	SG_REF(features);	
	features->start_parser();

	float64_t* fv;
	int32_t label, length, vector_number = 0;
	
	while (features->get_next_feature_vector(fv, length, label))
	{
		vector_number++;
		
		printf("Vector %d (Reading)\n--------------\n", vector_number);
		for (int i=0; i<length; i++)
		{
			printf("Fv[%d] = %f\t", i, fv[i]);

		}
		printf("Length = %d\t", length);
		printf("Label = %d\n\n", label);
	}
	
	
	features->end_parser();
	SG_UNREF(features);
	
	exit_shogun();
	return 0;
}
