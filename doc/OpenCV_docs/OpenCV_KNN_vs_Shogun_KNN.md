## k-Nearest Neighbours comparison between Shogun and OpenCV.

We will do a between Shogun's and OpenCV's k-NN implementations using a standard multi-class data-set available [here.](http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data). Our dataset consists of 1728 examples in which we will use the first half (864) as the training data and the rest as the testing data.

Let's start with the includes!
```CPP
// Shogun includes.
#include <shogun/base/init.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/OpenCV/CV2SGFactory.h>
#include <shogun/features/DataGenerator.h>

// OpenCV includes.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

// STL includes.
#include <iostream>
```

Now the namespaces.
```CPP
using namespace std;
using namespace shogun;
using namespace cv;

```


Here comes the actual benchmarking code!
```CPP
#define k 10 // The number of neighbours.

int main()
{
    init_shogun_with_defaults();
```

We will be using the CvMLData class of OpenCV.
```CPP
    CvMLData mlData;
    mlData.read_csv("car.data");
```

The data have the class response (outcome) written as the last index of each row.

We get a pointer to a ```CvMat``` object containing all the data. The total number of features is ```the total number columns - 1```.

```CPP
    const CvMat* temp = mlData.get_values();
    int numfeatures = temp->cols-1;
    mlData.set_response_idx(numfeatures);
```

We divide the data available into two equal parts. The first half is used for training and the rest for testing.
```CPP
    CvTrainTestSplit spl((float)0.5);
    mlData.set_train_test_split(&spl);
```

We get the respective indices of the training and testing data and store it in the ```cv::Mat``` format.
```CPP
    const CvMat* traindata_idx = mlData.get_train_sample_idx();
    const CvMat* testdata_idx = mlData.get_test_sample_idx();
    Mat mytraindataidx(traindata_idx);
    Mat mytestdataidx(testdata_idx);
```

We declare a few ```cv::Mat``` objects below, they will be used later use for our work.
* ```all_Data```: for containing the whole matrix offered to us by the ```.data``` file. 
* ```all_responses```: for containing all the responses.
* ```shogun_all_responses```: for containing all the responses for **Shogun**.
* ```traindata```: for containing all the training data.
* ```shogun_trainresponse```: for containing all the outputs we are provided for the training data as needed by **Shogun** for carrying out multiclass classification.
* ```opencv_trainresponse```: for containing all the outputs we are provided for the training data as needed by **OpenCV** for carrying out multiclass classification.
* ```testdata```: for containing all the testing data.
* ```shogun_testresponse```: for containing all the outputs of the test data(for **Shogun**). This will be used for evaluation purpose.
* ```opencv_testresponse```: for containing all the outputs of the test data(for **OpenCV**). This will be used for evaluation purpose.


```CPP
    Mat all_Data(temp);
    Mat all_responses = mlData.get_responses();
    Mat traindata(mytraindataidx.cols,numfeatures,CV_32F);
    Mat testdata(mytestdataidx.cols,numfeatures,CV_32F);
    Mat shogun_trainresponse(mytraindataidx.cols,1,CV_32S);
    Mat shogun_testresponse(mytestdataidx.cols,1,CV_32S);
    Mat shogun_all_responses = Mat::ones(all_responses.rows, 1, CV_32F);
    Mat opencv_trainresponse(mytraindataidx.cols,1,CV_32S);
    Mat opencv_testresponse(mytestdataidx.cols,1,CV_32S);
```

```CPP

    // Making responses compatible to Shogun.
    for (int h=0; h<all_responses.rows; h++)
    {
        if (all_responses.at<float>(h) == 4 )
            shogun_all_responses.at<float>(h)=0;
        else if (all_responses.at<float>(h) == 10)
            shogun_all_responses.at<float>(h)=1;
        else if (all_responses.at<float>(h) == 11)
            shogun_all_responses.at<float>(h)=2;
        else 
            shogun_all_responses.at<float>(h)=3;
    }

```

```CPP
// Filling in shogun_testresponse, shogun_trainresponse, opencv_testresponse, opencv_trainresponse, 
// traindata, and testdata mats.
   
   for(int i=0; i<mytraindataidx.cols; i++)
    {
        opencv_trainresponse.at<int>(i)=all_responses.at<float>(mytraindataidx.at<int>(i));
        shogun_trainresponse.at<int>(i)=shogun_all_responses.at<float>(mytraindataidx.at<int>(i));    
        for(int j=0; j<=numfeatures; j++)
            traindata.at<float>(i, j)=all_Data.at<float>(mytraindataidx.at<int>(i), j);
    }

    for(int i=0; i<mytestdataidx.cols; i++)
    {
        opencv_testresponse.at<int>(i)=all_responses.at<float>(mytestdataidx.at<int>(i));
        shogun_testresponse.at<int>(i)=shogun_all_responses.at<float>(mytestdataidx.at<int>(i));
        for(int j=0; j<=numfeatures; j++)
            testdata.at<float>(i, j)=all_Data.at<float>(mytestdataidx.at<int>(i), j);
    }
```

We train the **OpenCV** k-NN over the ```traindata``` Mat we just prepared.
```CPP
    CvKNearest opencv_knn(traindata, opencv_trainresponse);
    opencv_knn.train(traindata, opencv_trainresponse);
```
We test the trained model over the ```testdata``` Mat. 

Then, evaluate the accuracy using the ```opencv_trainresponse``` Mat.
```CPP
    Mat results(1,1,CV_32F);
    Mat neighbourResponses = Mat::ones(1,10,CV_32F);
    Mat dist = Mat::ones(1, 10, CV_32F);
 
    int ko=0;

    for (int i=0;i<testdata.rows;++i)
    {
        opencv_knn.find_nearest(testdata.row(i),10,results, neighbourResponses, dist);
        if (results.at<float>(0,0) == opencv_testresponse.at<int>(i))
            ++ko;
    }

    cout << "The accuracy of OpenCV's k-NN is: " << 100.0 * ko/testdata.rows << endl;
```

We, as usual, prepare the ```CDenseFeatures``` object namely ```shogun_trainfeatures``` for training the **Shogun** k-NN over it. 
```CPP

    SGMatrix<float64_t> shogun_traindata = CV2SGFactory::get_sgmatrix<float64_t>(traindata);
    SGMatrix<float64_t>::transpose_matrix(shogun_traindata.matrix, 
    		shogun_traindata.num_rows, shogun_traindata.num_cols);
    CDenseFeatures<float64_t>* shogun_trainfeatures = new CDenseFeatures<float64_t>(shogun_traindata);
```

We form the ```CMulticlassLabels``` object named ```labels``` for containing the responses from the ```shogun_trainresponse``` Mat.
```CPP
    CDenseFeatures<float64_t>* shogun_dense_response = 
    		CV2SGFactory::get_dense_features<float64_t>(shogun_trainresponse);
    SGVector<float64_t> shogun_vector_response = shogun_dense_response->get_feature_vector(0);
    CMulticlassLabels* labels = new CMulticlassLabels(shogun_vector_response);
```

We, as usual, prepare the ```CDenseFeatures``` object namely ```shogun_testfeatures``` for testing. 
```CPP
    SGMatrix<float64_t> shogun_testdata = CV2SGFactory::get_sgmatrix<float64_t>(testdata);
    SGMatrix<float64_t>::transpose_matrix(shogun_testdata.matrix,
    		shogun_testdata.num_rows, shogun_testdata.num_cols);
    CDenseFeatures<float64_t>* shogun_testfeatures = new CDenseFeatures<float64_t>(shogun_testdata);
```
___
**Shogun's** k-NN implementation.
```CPP
    // Create k-NN classifier.
	CKNN* knn = new CKNN(k, new CEuclideanDistance(shogun_trainfeatures, shogun_trainfeatures), labels);

	// Train classifier.
	knn->train();
```

Test it!
```CPP
    CMulticlassLabels* output = knn->apply_multiclass(shogun_testfeatures);
    SGMatrix<int32_t> multiple_k_output = knn->classify_for_multiple_k();
    SGVector<float64_t> sgvec = output->get_labels();

    int ki=0;
    for(int i=0; i<sgvec.vlen; ++i)
    { 
        if(shogun_testresponse.at<float>(i) == sgvec[i])
        ++ki;
    }

    cout << "The accuracy of Shogun's k-NN is: " << (float)100.0 *ki/sgvec.vlen << endl;
 	
	SG_UNREF(knn)
	SG_UNREF(output)  

    return 0;
}

```
OUTPUT
```
The accuracy of OpenCV's k-NN is: 79.7454.
The accuracy of Shogun's k-NN is: 66.5509.
```
