#include <shogun/lib/config.h>
#ifdef HAVE_OPENCV

#ifndef __CV2SGMATRIXFACTORY_H__
#define __CV2SGMATRIXFACTORY_H__


#include <opencv2/highgui/highgui.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun{

enum CV2SGOptions {CV2SG_CONSTRUCTOR, CV2SG_MANUAL, CV2SG_MEMCPY};

template <typename T> struct CV2SGTypeName;

template<> struct CV2SGTypeName<uint8_t>
{
	static const int get(){return 0;}
};

template<> struct CV2SGTypeName<int8_t>
{
	static const int get(){return 1;}
};

template<> struct CV2SGTypeName<uint16_t>
{
	static const int get(){return 2;}
};

template<> struct CV2SGTypeName<int16_t>
{
	static const int get(){return 3;}
};

template<> struct CV2SGTypeName<int32_t>
{
	static const int get(){return 4;}
};

template<> struct CV2SGTypeName<float32_t>
{
	static const int get(){return 5;}
};

template<> struct CV2SGTypeName<float64_t>
{
    static const int get(){return 6;}
};

class CV2SGMatrixFactory
{
	public:
	
	CV2SGMatrixFactory();
	
	virtual ~CV2SGMatrixFactory();
	
	template <typename T> static SGMatrix<T> getMatrix(cv::Mat, CV2SGOptions=CV2SG_CONSTRUCTOR);
};

template<typename T> SGMatrix<T> CV2SGMatrixFactory::getMatrix(cv::Mat cvMat, CV2SGOptions option)
{
	int nRows=cvMat.rows;
	int nCols=cvMat.cols;
	const int myType = CV2SGTypeName<T>::get();
	cvMat.convertTo(cvMat,myType);
	switch (option)
	{

		case CV2SG_CONSTRUCTOR:
		{
			SGMatrix<T> sgMat((T*)cvMat.data, nRows, nCols, false);
			return sgMat;
		}

		case CV2SG_MANUAL:
		{
			SGMatrix<T> sgMat(nRows, nCols);

			for(int i=0; i < nRows; i++)
				for(int j=0; j < nCols; j++)
					sgMat(i,j) = cvMat.at<T>(i,j);
			return sgMat;
		}

		case CV2SG_MEMCPY:
		{
			SGMatrix<T> sgMat(nRows, nCols);
			memcpy(sgMat.matrix, cvMat.data, nRows*nCols*sizeof(T));
			SGMatrix<T>::transpose_matrix(sgMat.matrix,nRows,nCols);
			return sgMat;
		}
		
		default:
		{
			SGMatrix<T> sgMat(nRows, nCols);
			memcpy(sgMat.matrix, cvMat.data, nRows*nCols*sizeof(T));
			SGMatrix<T>::transpose_matrix(sgMat.matrix,nRows,nCols);
			return sgMat;
		}
	}
}

}
#endif /*__CV2SGMATRIXFACTORY_H__*/
#endif /* HAVE_OPENCV */
