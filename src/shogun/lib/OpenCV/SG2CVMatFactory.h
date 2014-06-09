/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Abhijeet Kislay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/config.h>
#ifdef HAVE_OPENCV

#ifndef SG2_CVMAT_FACTORY_H_
#define SG2_CVMAT_FACTORY_H_

#include <opencv2/highgui/highgui.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/OpenCV/OpenCVTypeName.h>

namespace shogun{

enum SG2CVOptions {SG2CV_CONSTRUCTOR, SG2CV_MANUAL, SG2CV_MEMCPY};


class SG2CVMatFactory
{

private:
    template<typename T1, typename T2> static cv::Mat getMatUsingManual(SGMatrix<T1>, int, int, int, int);
    template<typename T1, typename T2> static cv::Mat getMatUsingMemcpy(SGMatrix<T1>, int, int, int, int);
    template<typename T1, typename T2> static cv::Mat getMatUsingConstructor(SGMatrix<T1>, int, int, int, int);

public:
	SG2CVMatFactory();		
	~SG2CVMatFactory();
	template <typename T1, typename T2> static cv::Mat getcvMat(CDenseFeatures<T1>* A,
			SG2CVOptions=SG2CV_MEMCPY);
};

template<typename T1, typename T2> cv::Mat SG2CVMatFactory::getMatUsingManual(SGMatrix<T1> sgMat, int num_rows, int num_cols, int myType, int initType)
{
    cv::Mat cvMat(num_rows, num_cols, myType);
	for(int i=0; i<num_rows; i++)
	{
		for(int j=0; j<num_cols; j++)
		{
			cvMat.at<T2>(i, j)=sgMat(i, j);
		}
	}
	return cvMat;
}

template<typename T1, typename T2> cv::Mat SG2CVMatFactory::getMatUsingMemcpy(SGMatrix<T1> sgMat, int num_rows, int num_cols, int myType, int initType)
{
    cv::Mat cvMat(num_rows, num_cols, initType);
	memcpy((T1*)cvMat.data, sgMat.matrix, num_rows*num_cols*sizeof(T1));
	cvMat.convertTo(cvMat,myType);
	return cvMat;

}

template<typename T1, typename T2> cv::Mat SG2CVMatFactory::getMatUsingConstructor(SGMatrix<T1> sgMat, int num_rows, int num_cols, int myType, int initType)
{
    cv::Mat cvMat(num_rows, num_cols, initType, (void*)sgMat.matrix);
	cvMat.convertTo(cvMat, myType);
	return cvMat;

}

template <typename T1, typename T2> cv::Mat SG2CVMatFactory::getcvMat(CDenseFeatures<T1>* A, SG2CVOptions option)
{
	SGMatrix<T1> sgMat=((CDenseFeatures<T1>*)A)->get_feature_matrix();
	int num_rows=sgMat.num_rows;
	int num_cols=sgMat.num_cols;
	const int myType=OpenCVTypeName<T2>::get_opencv_type();
	const int initType=OpenCVTypeName<T1>::get_opencv_type();

	switch(option)
	{
		case SG2CV_MEMCPY:
		{
            cv::Mat cvMat = SG2CVMatFactory::getMatUsingMemcpy<T1, T2>(sgMat, num_rows, num_cols, myType, initType); 
            return cvMat;
		}
		
		case SG2CV_MANUAL:
		{
            cv::Mat cvMat = SG2CVMatFactory::getMatUsingManual<T1, T2>(sgMat, num_rows, num_cols, myType, initType); 
            return cvMat;
        }

		case SG2CV_CONSTRUCTOR:
		{
            cv::Mat cvMat = SG2CVMatFactory::getMatUsingConstructor<T1, T2>(sgMat, num_rows, num_cols, myType, initType); 
            return cvMat;
        }
		
		default:
		{
            cv::Mat cvMat = SG2CVMatFactory::getMatUsingMemcpy<T1, T2>(sgMat, num_rows, num_cols, myType, initType); 
            return cvMat;
	    }
	}

}

}

#endif /*SG2_CVMAT_FACTORY_H_*/
#endif /* HAVE_OPENCV */
