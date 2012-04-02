#include <shogun/features/BOWFeatures.h>
#include <shogun/base/init.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/features/Labels.h>


using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}
SGMatrix<int32_t> get_bagofwords(CSimpleFeatures<float64_t>* features, int32_t num_clusters, int32_t dim_descriptor);

const int32_t num_vectors=6;
const int32_t dim_features=4;
const int32_t dim_descriptor=2;
const int32_t num_clusters=4; 	

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);
	//const int32_t num_subset_idx=CMath::random(1, num_vectors);


	/* create feature data matrix */
	SGMatrix<float64_t> data(dim_features,num_vectors);

	/* fill matrix with random data of unequal feature sizes */
	for (index_t i=0; i<num_vectors; ++i)
	{	
		for (index_t j=0; j<dim_features; ++j)
		{
			data.matrix[i*dim_features+j]=CMath::random(-5, 5);
			if(i>3 && j>3) data.matrix[i*dim_features+j]=0.00;
			
		}	
	}

	/* create simple features */

	CSimpleFeatures<float64_t>* features=new CSimpleFeatures<float64_t> (data);
	SG_REF(features);

	/* print input feature matrix */
	CMath::display_matrix(data.matrix, data.num_rows, data.num_cols, "input feature matrix");
	
	SGMatrix<int32_t> histogram(get_bagofwords(features, num_clusters, dim_descriptor));

	/* print output feature matrix */		
	CMath::display_matrix(histogram.matrix, num_clusters, num_vectors, "output feature matrix");
	
	
	

	SG_UNREF(features);
	SG_SPRINT("\nEND\n");

	exit_shogun();

	
	return 0;

	
}
	/* this point on should go into BOWFeatures.h*/
	
	/* input feature, dim_descriptor, num_clusters*/
	
	/* feature to descriptor set */
SGMatrix<int32_t> get_bagofwords(CSimpleFeatures<float64_t>* features, int32_t num_clusters, int32_t dim_descriptor)
{

	SGMatrix<float64_t> data2(features->get_feature_matrix());
	
	int32_t num_descriptor=data2.num_rows*data2.num_cols/dim_descriptor;
	SGMatrix<float64_t> data_descriptor(dim_descriptor,num_descriptor);

	index_t count=0;
        for (index_t i=0; i<dim_descriptor; ++i)
        {
                for (index_t j=0; j<num_descriptor; ++j)
                {
                        data_descriptor.matrix[i*num_descriptor+j]=data2.matrix[count];
			count++;
                }
        }
        
	/* create shogun feature for the descriptors*/
        CSimpleFeatures<float64_t>* descriptors=new CSimpleFeatures<float64_t> (data2);
        SG_REF(descriptors);
 
 	CMath::display_matrix(data_descriptor.matrix, data_descriptor.num_rows, data_descriptor.num_cols,
                        "descriptors from features");


        int32_t num_features=num_vectors;
        int32_t num_vectors_per_cluster=5;
        float64_t cluster_std_dev=2.0;

        /* build random cluster centers */
	
  	CEuclidianDistance* distance=new CEuclidianDistance(descriptors, descriptors);
        CKMeans* clustering=new CKMeans(num_clusters, distance);
        clustering->train(descriptors);
	/*  the visual vocabulary */
	       /* build clusters */

        CLabels* result=clustering->apply();
        for (index_t i=0; i<result->get_num_labels(); ++i)
                SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));


	SGMatrix<float64_t> cluster_centers(clustering->get_cluster_centers());	
	 CMath::display_matrix(cluster_centers.matrix, dim_descriptor, num_clusters,
                        "cluster centers");
	/* quantizing feature */

	index_t minj=0, tempi=0;
	float64_t minv=0, tempv=0;
	
	 SGMatrix<int32_t> quantize(num_descriptor,1);
	for(index_t i=0; i<num_descriptor; i++)
	{
			minj=0;
			//minv=distmatrix(0,i);
			minv=100;
		for(index_t j=0;j<num_clusters;j++)
		{
			tempv=0;
		
			for(index_t k=0; k<dim_descriptor;k++)
			{
tempv = tempv + CMath::pow((cluster_centers[j*dim_descriptor + k ] - data_descriptor[i*dim_descriptor + k] ),2);
			 	
			}

			if(tempv<minv)
			{
				minv=tempv;
				minj=j;
			}
			
		}		
			quantize[i]= minj;
		
	}

   CMath::display_matrix(quantize.matrix, num_descriptor, 1, "quantization matrix");

 /*  histogram creation . and returning the histogram out as the feature*/
 /* dimension .. > num_features num_clsuter*/

	SGMatrix<int32_t> histogram(num_clusters,num_features);
	int32_t ratio; 
	int32_t value;
	for(index_t i=0;i<num_features;i++)
		for(index_t j=0; j<num_clusters; j++)
			histogram[i*num_clusters +j]=0;
	ratio = dim_features/dim_descriptor;
	count=0;
	for(index_t i=0;i<num_features; i ++)
	{
		for(index_t j=0; j<ratio;j++)
		{	
			value=quantize[count++];
			histogram[ i*num_clusters + value ]++;
			
			
		}
		
	}
	
	cluster_centers.destroy_matrix();

        SG_UNREF(result);
	SG_UNREF(descriptors);

  
	return histogram;

}
