###Neural network comparison between Shogun and OpenCV

In this document, we will do a comparison between Shogun's implementation of neural networks and OpenCV's one using a standard multi-class data-set available [here.](http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data) Our dataset consists of 1728 examples. We will use the first half (864) as the training data and the rest as the testing data.

Let's start with the includes!
```CPP
// shogun includes.
#include <shogun/base/init.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/lib/OpenCV/CV2SGFactory.h>

// standard library.
#include <iostream>

// opencv includes.
#include<opencv2/ml/ml.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

// for measuring time.
#include <omp.h>
// The variable start will be later used in the time measurement calculations.
double start;
#define ntime start=omp_get_wtime()
#define ftime <<omp_get_wtime()-start<<endl
```

Now the namespaces.
```CPP
using namespace shogun;
using namespace std;
using namespace cv;

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

The data have the class response (outcome) written as the last index of each row.

We get a pointer to a ```CvMat``` object containing all the data. Total number of the features is ```the total columns - 1```.

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

We get the respective indices of the training and testing data and store it in the ```cv::Mat``` format.
```CPP

    const CvMat* traindata_idx = mlData.get_train_sample_idx();
    const CvMat* testdata_idx = mlData.get_test_sample_idx();
    Mat mytraindataidx(traindata_idx);
    Mat mytestdataidx(testdata_idx);
```

We declare few cv::Mat objects down there which we will later use .
* ```all_Data```: for containing the whole matrix offered to us by the ```.data``` file. 
* ```all_responses```: for containing all the responses.
* ```opencv_all_responses```: for containing all the responses for **OpenCV**.
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
    Mat shogun_trainresponse(mytraindataidx.cols,1,CV_32S);
    Mat opencv_trainresponse(mytraindataidx.cols,4,CV_32F);
    Mat testdata(mytestdataidx.cols,numfeatures,CV_32F);
    Mat shogun_testresponse(mytestdataidx.cols,1,CV_32S);
    Mat opencv_testresponse(mytestdataidx.cols,4,CV_32F);
    Mat opencv_all_responses = Mat::ones(all_responses.rows, 4, CV_32F);
    Mat shogun_all_responses = Mat::ones(all_responses.rows, 1, CV_32F);

```

As for now **OpenCV** doesnot support multiclass responses. The workaround is to create a binary tuple of ```M``` elements.( Here ```M``` is number of classes whose value is greater than ```2``` ).

Hence here we create ```4``` tuples, each one for a separate class.
```CPP
   
    float data1[]={1,0,0,0};
    float data2[]={0,1,0,0};
    float data3[]={0,0,1,0};
    float data4[]={0,0,0,1};

    Mat data1Mat(1,4,CV_32F,data1);
    Mat data2Mat(1,4,CV_32F,data2);
    Mat data3Mat(1,4,CV_32F,data3);
    Mat data4Mat(1,4,CV_32F,data4);

```

We fill in the responses from ```all_responses``` to the two respective response Mat objects of **OpenCV** and **Shogun** namely ```opencv_all_responses``` and ```shogun_all_responses```.

```CPP
    for (int h=0; h<all_responses.rows; h++)
    {
        if (all_responses.at<float>(h) == 4 )
        {
            data1Mat.copyTo(opencv_all_responses.row(h));
            shogun_all_responses.at<float>(h)=0;
        }
        else if (all_responses.at<float>(h) == 10)
        {
            data2Mat.copyTo(opencv_all_responses.row(h));
            shogun_all_responses.at<float>(h)=1;
        }
        else if (all_responses.at<float>(h) == 11)
        {
            data3Mat.copyTo(opencv_all_responses.row(h));
            shogun_all_responses.at<float>(h)=2;
        }
        else 
        {
            data4Mat.copyTo(opencv_all_responses.row(h));
            shogun_all_responses.at<float>(h)=3;
        }
    }
```

We fill in the ```traindata ```,  ```testdata```, ```opencv_train_response```, ```shogun_train_response```, ```opencv_test_response```, ```shogun_test_response``` Mats which were defined above.

```CPP

   for(int i=0; i<mytraindataidx.cols; i++)
    {
        opencv_all_responses.row(mytraindataidx.at<int>(i)).copyTo(opencv_trainresponse.row(i));
        shogun_trainresponse.at<int>(i)=shogun_all_responses.at<float>(mytraindataidx.at<int>(i));    
        for(int j=0; j<=numfeatures; j++)
        {
            traindata.at<float>(i, j)=all_Data.at<float>(mytraindataidx.at<int>(i), j);
        }
    }

    for(int i=0; i<mytestdataidx.cols; i++)
    {
        opencv_all_responses.row(mytestdataidx.at<int>(i)).copyTo(opencv_testresponse.row(i));
        shogun_testresponse.at<int>(i)=shogun_all_responses.at<float>(mytestdataidx.at<int>(i));
        for(int j=0; j<=numfeatures; j++)
        {
            testdata.at<float>(i, j)=all_Data.at<float>(mytestdataidx.at<int>(i), j);
        }   
    }

```

Here I have created a 3 layered network. The input layer consists of 6 neurons which is equal to number of features. The hidden layer has 10 neurons and similarly the output layer has 4 neurons which is equal to the number of classes.

```CPP
    int layersize_array[] = {6,10,4};
    Mat layersize_mat(1,3,CV_32S,layersize_array);

    CvANN_MLP_TrainParams cvtrainparams = CvANN_MLP_TrainParams(cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000,  1e-8 ), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);

    CvANN_MLP neural_network = CvANN_MLP();
    neural_network.create(layersize_mat ,CvANN_MLP::SIGMOID_SYM);
```

Train it!

```CPP
    ntime;
    neural_network.train(traindata, opencv_trainresponse, Mat());
    ftime;
```

Test it!

```CPP
    Point p_max, test_max;

    int k=0;
    Mat predicted_tuple(1,4, CV_32F);
    for (int i=0; i<testdata.rows; ++i)
    { 
        neural_network.predict(testdata.row(i), predicted_tuple);
        minMaxLoc(predicted_tuple,NULL,NULL,NULL,&p_max);
        minMaxLoc(opencv_testresponse.row(i),NULL, NULL, NULL, &test_max);
        if (p_max.x == test_max.x)
        ++k;
    }
    cout<< "Our OpenCV NN gives an accuracy of: "<< 100.0* k/testdata.rows<<endl;
```

Now we start with the Neural Network implementation in **Shogun**.

As usual, we start with creating a ```DenseFeatures``` object with the training data.

```CPP
    SGMatrix<float64_t> shogun_traindata = CV2SGFactory::get_sgmatrix<float64_t>(traindata);
    SGMatrix<float64_t>::transpose_matrix(shogun_traindata.matrix, shogun_traindata.num_rows, shogun_traindata.num_cols);
    CDenseFeatures<float64_t>* shogun_trainfeatures = new CDenseFeatures<float64_t>(shogun_traindata);
```


The training responses are in an object of type ```MulticlassLabels```.
```CPP
    CDenseFeatures<float64_t>* shogun_dense_response = CV2SGFactory::get_dense_features<float64_t>(shogun_trainresponse);
    SGVector<float64_t> shogun_vector_response = shogun_dense_response->get_feature_vector(0);
    CMulticlassLabels* labels = new CMulticlassLabels(shogun_vector_response);
```


Prepare the testing data.
```CPP
    SGMatrix<float64_t> shogun_testdata = CV2SGFactory::get_sgmatrix<float64_t>(testdata);
    SGMatrix<float64_t>::transpose_matrix(shogun_testdata.matrix, shogun_testdata.num_rows, shogun_testdata.num_cols);
    CDenseFeatures<float64_t>* testfeatures = new CDenseFeatures<float64_t>(shogun_testdata);
```


To use Neural Networks in **Shogun** the following things need to be done:

* Prepare a ```CDynamicObjectArray``` of ```CNeuralLayer```-based objects that specify the type of layers used in the network. The array must contain at least one input layer. The last layer in the array is treated as the output layer. Also note that forward propagation is performed in the order the layers appear in the array. So, if layer ```j``` takes its input from layer ```i```, then ```i``` must be less than ```j```.

* Specify how the layers are connected together. This can be done using either ```connect()``` or ```quick_connect()```.

* Call ```initialize()```.

* Specify the training parameters if needed.

* Train ```set_labels()``` and ```train()```.

* If needed, the network with the learned parameters can be stored on disk using ```save_serializable()``` ( loaded using ```load_serializable()```)

* Apply the network using ```apply()```.


---
* Let us start with the first step.

We will be preparing a ```CDynamicObjectArray```. It creates an array that can be used like a list or an array.
We then append information related to the number of neurons per layer in the respective order.

Here I have created a ```3``` layered network. The input layer consists of ```6``` neurons which is equal to number of features.
The hidden layer has ```10``` neurons and similarly the output layer has ```4``` neurons which is equal to the number of classes.

```CPP
    CDynamicObjectArray* layers = new CDynamicObjectArray();
    layers->append_element(new CNeuralInputLayer(6));
    layers->append_element(new CNeuralLogisticLayer(10)); 
    layers->append_element(new CNeuralLogisticLayer(4));
```

* Here we have to make a connection between the three layers that we formed above. To connect each neuron of one layer to each one of the layer suceeding it, we can directly use ```quick_connect()```. However If particular connections are to be made separately, we may have to use ```connect()```.  

```CPP
    CNeuralNetwork* network = new CNeuralNetwork(layers);
    network->quick_connect();
```

* Initialize the network. The input is nothing but the standard deviation of the gaussian which is used to randomly initialize the parameters. We chose ```0.1``` here.

```CPP
    network->initialize(0.1);
```

* Specify the training parameters if needed. 

```CPP
    network->epsilon = 1e-8;
    network->max_num_epochs = 1000;
```

* Set labels and train!

```CPP
    network->set_labels(labels);
    ntime;
    network->train(shogun_trainfeatures);
    ftime;
```

* Test it!

```CPP
    CMulticlassLabels* predictions = network->apply_multiclass(testfeatures);
    k=0;
    for (int32_t i=0; i<mytraindataidx.cols; i++ )
    {
        if (predictions->get_label(i)==shogun_testresponse.at<int>(i))
        ++k;
    }
    <<"Our Shogun NN gives an accuracy of: "<<100.0*k/(mytraindataidx.cols)<<endl;
    return 0;
}
```

Output!

1st time
```
    2.32288
    Our OpenCV NN gives an accuracy of: 68.8657
    0.39906
    Our Shogun NN gives an accuracy of: 81.713
```

2nd time
```
    2.33449
    Our OpenCV NN gives an accuracy of: 68.8657
    0.39428
    Our Shogun NN gives an accuracy of: 78.125
```

3rd time
```sh
    2.30646
    Our OpenCV NN gives an accuracy of: 68.8657
    0.40048
    Our Shogun NN gives an accuracy of: 76.8519
```
