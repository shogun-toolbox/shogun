###Support Vector Machines comparison between Shogun and OpenCV

We will try to do a one to one comparison between the Shogun's implementaton of LibSVM to that of OpenCV's one on a standard multi-class data-set available [here.](http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data) Our dataset consists of 1728 examples in which we will use the first half (864) as the training data and the rest as the testing data.

Let's start with the includes!

```CPP
//standard library
#include <iostream>

// opencv includes.
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// shogun includes.
#include <shogun/base/init.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/OpenCV/CV2SGFactory.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/kernel/LinearKernel.h>

// for measuring time
#include <omp.h>
// The variable start will be later used in the time measurement calculations.
double start;
#define ntime start=omp_get_wtime()
#define ftime cout<<omp_get_wtime()-start<<endl
```

Now the namespaces.
```
using namespace shogun;
using namespace cv;
using namespace std;
```

Here comes the actual benchmarking code!
```CPP
int main()
{
    init_shogun_with_defaults();
```

We will be using the ```CvMLData``` class of OpenCV.
```CPP
    CvMLData mlData;
    mlData.read_csv("car.data");
```

The data that we have has the class response(outcome) written as the last index of each row.

We get a pointer to a ```CvMat``` object containing all the data. Total number of the features is ```total columns -1```.

```CPP
    const CvMat* temp = mlData.get_values();
    int numfeatures = temp->cols-1;
    mlData.set_response_idx(numfeatures);

```

We divide the data available to us into two equal parts. The first half is used for training and the rest half for testing.
```CPP

    CvTrainTestSplit spl((float)0.5);
    mlData.set_train_test_split(&spl);
```

We get the respective indices of the training and testing data and store it in the cv::Mat format.
```CPP
	const CvMat* traindata_idx = mlData.get_train_sample_idx();
	const CvMat* testdata_idx = mlData.get_test_sample_idx();
	Mat mytraindataidx(traindata_idx);
	Mat mytestdataidx(testdata_idx);

```

We declare few cv::Mat objects down there which we will later use for our work.
* ```all_Data```: for containing the whole matrix offered to us by the ```.data``` file.
* ```all_responses```: for containing all the responses.
* ```traindata```: for containing all the training data.
* ```trainresponse```: for containing all the outputs we are provided for the training data.
* ```testdata```: for containing all the testing data.
* ```testresponse```: for containing all the outputs of the test data. This will be used for evaluation purpose.

```CPP
	Mat all_Data(temp);
	Mat all_responses = mlData.get_responses();
	Mat traindata(mytraindataidx.cols,numfeatures,CV_32F);
	Mat trainresponse(mytraindataidx.cols,1,CV_32S);
	Mat testdata(mytestdataidx.cols,numfeatures,CV_32F);
	Mat testresponse(mytestdataidx.cols,1,CV_32S);
```

We try to fill in the ```traindata```, ```testdata```,```trainresponse``` and ```testresponse``` Mats which were defined above.
```CPP
    for(int i=0; i<mytraindataidx.cols; i++)
    {
    	trainresponse.at<int>(i)=all_responses.at<float>(mytraindataidx.at<int>(i));
    	for(int j=0; j<=numfeatures; j++)
    	{
    		traindata.at<float>(i, j)=all_Data.at<float>(mytraindataidx.at<int>(i), j);
    	}
    }

    for(int i=0; i<mytestdataidx.cols; i++)
    {
    	testresponse.at<int>(i)=all_responses.at<float>(mytestdataidx.at<int>(i));
    	for(int j=0; j<=numfeatures; j++)
    	{
    		testdata.at<float>(i, j)=all_Data.at<float>(mytestdataidx.at<int>(i), j);
    	}
    }

```

With the traindata and trainresponse Mat ready, we too are ready for testing SVM on it.
```CPP
    CvSVM opencv_svm;
    CvSVMParams params;
    params.svm_type=SVM::C_SVC;
	params.C=0.1;
	params.kernel_type=SVM::LINEAR;
	params.term_crit=TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
```

Run it! Train it!
```CPP
    ntime;
	opencv_svm.train(traindata,trainresponse, Mat(), Mat(),params);
    ftime;
```

Testing procedure.

```
    int k=0;
    for(int i=0; i<testdata.rows; i++)
	{
		Mat samplemat(testdata, Range(i,i+1));
		float response = opencv_svm.predict(samplemat,numfeatures);
	    k=(response==testresponse.at<int>(i))?++k:k;
	}
    cout<<"accuracy by the opencv svm is ..." <<100.0 * (float)k/testdata.rows<<endl;
```

Now we will work on the training of Shogun Multiclass LibSVM on the same dataset. Since the working of the Shogun library is a bit different from that of OpenCV, we might need to tweak it a little.

As mentioned earlier, we have 4 classes in the dataset which are non-numerical in nature. This means that it's the job of OpenCV ```read_csv``` to convert these non-numerical responses into some numerics. For details read [official documentation](http://docs.opencv.org/modules/ml/doc/mldata.html#cvmldata-read-csv)

We here observe that the ```read_csv``` returns``` 4, 10, 11 ```and``` 12``` as the 4 different responses. We just convert it into ```0, 1, 2 ```and ```3``` as Shogun likes it this way only!

```CPP
    for (int h=0; h<all_responses.rows; h++)
    {
        if (all_responses.at<float>(h) == 4 )
            all_responses.at<float>(h)=0;

        else if (all_responses.at<float>(h) == 10)
            all_responses.at<float>(h)=1;

        else if (all_responses.at<float>(h) == 11)
            all_responses.at<float>(h)=2;

        else all_responses.at<float>(h)=3;
    }
```

We need to fill in the ```testresponse``` and ```trainresponse``` with the above converted responses.
```CPP
  for(int i=0; i<mytraindataidx.cols; i++)
    {
        trainresponse.at<int>(i)=all_responses.at<float>(mytraindataidx.at<int>(i));
    }

    for(int i=0; i<mytestdataidx.cols; i++)
    {
        testresponse.at<int>(i)=all_responses.at<float>(mytestdataidx.at<int>(i));
    }
```

We here start preparing for the SVM implementation in shogun. Things that we will be needing are:
* The training data in the form of ```DenseFeatures```.
* The training responses in the form of ```MulticlassLabels```.
* A kernel.

We start with creating the training data as the ```DenseFeatures```.

```CPP
    SGMatrix<float64_t> shogun_traindata = CV2SGFactory::get_sgmatrix<float64_t>(traindata);
    SGMatrix<float64_t>::transpose_matrix(shogun_traindata.matrix, shogun_traindata.num_rows, shogun_traindata.num_cols);
    CDenseFeatures<float64_t>* shogun_trainfeatures = new CDenseFeatures<float64_t>(shogun_traindata);
```

Now the training responses as the ```MulticlassLabels```.
```CPP
    CDenseFeatures<float64_t>* shogun_dense_response = CV2SGFactory::get_dense_features<float64_t>(trainresponse);
    SGVector<float64_t> shogun_vector_response = shogun_dense_response->get_feature_vector(0);
    CMulticlassLabels* labels = new CMulticlassLabels(shogun_vector_response);
```

Now the Kernel.

```CPP
    CLinearKernel* kernel = new CLinearKernel();
    kernel->init(shogun_trainfeatures, shogun_trainfeatures);
```

Now we are ready to initialize the SVM for Shogun. We train it here!
```CPP
    CMulticlassLibSVM* shogunsvm = new CMulticlassLibSVM(10, kernel, labels );
    ntime;
    shogunsvm->train();
    ftime;
```

Prepare the testing data.
```CPP
    SGMatrix<float64_t> shogun_testdata=CV2SGFactory::get_sgmatrix<float64_t>(testdata);
    SGMatrix<float64_t>::transpose_matrix(shogun_testdata.matrix, shogun_testdata.num_rows, shogun_testdata.num_cols);
    CDenseFeatures<float64_t>* testfeatures=new CDenseFeatures<float64_t>(shogun_testdata);
```

Testing Procedure.
```CPP
    CMulticlassLabels* results=shogunsvm->apply_multiclass(testfeatures);
    k=0;
    for(int i=0; i<testdata.rows; i++)
	{
        float response =  (results->get_labels()).get_element(i);
        float actual = testresponse.at<int>(i);
        k=(response==actual)?++k:k;
	}
    cout<<"accuracy by the shogun svm is"<<100.0 * (float)k/testdata.rows<<endl;

    exit_shogun();
    return 0;
}
```

Output!
```CPP
	150.366
	accuracy by the opencv svm is 77.0833
	0.215284
	accuracy by the shogun svm is 77.1991

```

We infer from the output that:
* The accuracy of OpenCV's LibSVM is 77.0833% with the time taken = 150.366 secs.
* The accuracy of Shogun's LibSVM is 77.1991% with the time taken = 0.215 secs.
