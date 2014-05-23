#ifdef HAVE_OPENCV

#include <opencv2/highgui/highgui.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun{


enum SG2CVOptions {SG2CV_CONSTRUCTOR, SG2CV_MANUAL, SG2CV_MEMCPY};

template <typename T> struct SG2CVTypeName;

template<> struct SG2CVTypeName<uint8_t>
{
	static const int get(){return 0;}
};

template<> struct SG2CVTypeName<int8_t>
{
	static const int get(){return 1;}
};

template<> struct SG2CVTypeName<uint16_t>
{
	static const int get(){return 2;}
};

template<> struct SG2CVTypeName<int16_t>
{
	static const int get(){return 3;}
};

template<> struct SG2CVTypeName<int32_t>
{
	static const int get(){return 4;}
};

template<> struct SG2CVTypeName<float32_t>
{
	static const int get(){return 5;}
};

template<> struct SG2CVTypeName<float64_t>
{
    static const int get(){return 6;}
};

class SG2CVMatFactory
{
public:
	
	SG2CVMatFactory();	
	
	virtual ~SG2CVMatFactory();
	
	template <typename T> static cv::Mat getMatrix(CDenseFeatures<float64_t>* A,
			SG2CVOptions=SG2CV_CONSTRUCTOR);
};

template <typename T> cv::Mat SG2CVMatFactory::getMatrix(CDenseFeatures<float64_t>* A,
		SG2CVOptions option)
{
	SGMatrix<float64_t> sgMat=((CDenseFeatures<float64_t>*)A)->get_feature_matrix();
	int nRows=sgMat.num_rows;
	int nCols=sgMat.num_cols;
	const int myType = SG2CVTypeName<T>::get();

	switch(option)
	{
		case SG2CV_CONSTRUCTOR:
		{
			cv::Mat cvMat(nRows, nCols, CV_64FC1, (void*)sgMat.matrix);
			cvMat.convertTo(cvMat,myType);
			return cvMat;
		}
		
		case SG2CV_MANUAL:
		{
			cv::Mat cvMat(nRows, nCols, CV_64FC1);
			for(int i=0; i<nRows; i++)
				for(int j=0; j<nCols;j++)
					cvMat.at<double>(i,j) = sgMat(i,j);
			cvMat.convertTo(cvMat,myType);
			return cvMat;
		}

		case SG2CV_MEMCPY:
		{
			cv::Mat cvMat(nRows, nCols, CV_64FC1);
			memcpy((double*)cvMat.data, sgMat.matrix, nRows*nCols*sizeof(double));
			cvMat.convertTo(cvMat,myType);
			return cvMat;
		}
	}
}

}
#endif /* HAVE_OPENCV */
