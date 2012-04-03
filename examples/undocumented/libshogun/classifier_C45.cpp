
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/Labels.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/classifier/C45.h>

#include <stdio.h>
#include <string.h>

using namespace shogun;

#define LINE_BYTES_MAX 1024

//the num of training samples and the dimension of samples
#define NUM 14
#define DIM 4

int testing_num=14;//the num of testing samples

float64_t* lab;
float64_t* feat;
float64_t information;

int32_t flag[DIM*2]= {	1,3, //reprensents attribute 0 is a discrete-valued attribute and it has 3 different values
                        1,3,//reprensents attribute 1 is a discrete-valued attribute and it has 3 different values
                        0,0,//reprensents attribute 2 is a continuous-valued attribute
                        0,0//reprensents attribute 3 is a continuous-valued attribute
                     };

typedef const char *string_a;

string_a values0[]= {"low","middle","high"};
string_a values1[]= {"small","medium","big"};
string_a *values[DIM];//values of discrete-valued attributes


int getIntByValue(const char* s,int index)
{
    for(int i=0; i<flag[index*2+1]; i++)
    {
        if (strcmp(s,values[index][i])==0)
        {
            return i;
        }
    }
    return -1;
}


/* the format of data file: (class_label:0/1)
 *
 * class_label	attr0	attr1	attr2	attr3	...
 * class_label	attr0	attr1	attr2	attr3	...
 * class_label	attr0	attr1	attr2	attr3	...
 * class_label	attr0	attr1	attr2	attr3	...
 * class_label	attr0	attr1	attr2	attr3	...
 * 		.
 * 		.
 * 		.
 */
void read_samples(FILE* f)
{
    int i=0,j=0;
    int total_yes_number=0,total_no_number=0;
    char *rst;
    char buffer[LINE_BYTES_MAX+1];
    lab=SG_MALLOC(float64_t,NUM);
    feat=SG_MALLOC(float64_t,NUM*DIM);

    values[0]=values0;//possible values of discrete-valued attribute0
    values[1]=values1;//possible values of discrete-valued attribute1
    values[2]=NULL;
    values[3]=NULL;

    while(i<NUM)
    {
        if(fgets(buffer,LINE_BYTES_MAX,f)==NULL)
        {
            break;
        }
        rst=strtok(buffer,"	\n");
        if((lab[i]=atoi(rst))==1)//classLAB
        {
            total_yes_number++;
        }

        rst=strtok(NULL,"	\n");
        j=0;
        while (rst!=NULL)
        {
            if(flag[j*2]==0)//continuous attribute
            {
                feat[i*DIM+j]=atoi(rst);
            }
            else
            {
                feat[i*DIM+j]=getIntByValue(rst,j);
            }


            rst=strtok(NULL,"	\n");
            j++;
        }
        i++;
    }

    float64_t log2=log(2.0);
    //compute the information
    total_no_number=NUM-total_yes_number;
    float64_t t2;
    t2=total_no_number/(float64_t)NUM;
    information=t2*log(t2)/log2;
    t2=total_yes_number/(float64_t)NUM;
    information+=(t2*log(t2)/log2);
    information=-information;
}


int main(int argc, char** argv)
{
    init_shogun_with_defaults();

    FILE* f=fopen("data.txt","r");
    read_samples(f);
    fclose(f);

    //create train labels
    CLabels* labels=new CLabels(SGVector<float64_t>(lab, NUM));
    SG_REF(labels);

    //create train features
    CSimpleFeatures<float64_t>* features=new CSimpleFeatures<float64_t>();
    SG_REF(features);
    features->set_feature_matrix(feat, DIM, NUM);

    //create Dicision Tree via C4.5 and train
    CC45* tree= new CC45(features,labels);
    SGMatrix<int32_t> attr_flag(flag,DIM,2);

    tree->set_attribute_list(attr_flag);
    tree->set_info(information);
    tree->train_machine();

    // classify on training examples
    for (int32_t i=0; i<NUM; i++)
        printf("original_label[%d]=%f	output[%d]=%f\n", i,lab[i],i, tree->apply(i));

    return 0;
}


