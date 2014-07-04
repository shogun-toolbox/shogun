/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Abhijeet Kislay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *	  list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *	  this list of conditions and the following disclaimer in the documentation
 *	  and/or other materials provided with the distribution.
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

#ifndef CV2_SGMATRIX_FACTORY_H_
#define CV2_SGMATRIX_FACTORY_H_

#include <opencv2/highgui/highgui.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/OpenCV/OpenCVTypeName.h>

namespace shogun
{
enum CV2SGOptions {CV2SG_CONSTRUCTOR, CV2SG_MANUAL, CV2SG_MEMCPY};

class CV2SGFactory
{
	public:
		CV2SGFactory();
		~CV2SGFactory();
		template<typename SG_T> static SGMatrix<SG_T> getSGMatrix(cv::Mat,
				CV2SGOptions=CV2SG_MEMCPY);
		
		template<typename SG_T> static CDenseFeatures<SG_T>* getDensefeatures(cv::Mat,
				CV2SGOptions=CV2SG_MEMCPY);

	private:
		template<typename SG_T> static SGMatrix<SG_T> getMatrixUsingManual
			(cv::Mat, int, int);

		template<typename SG_T> static SGMatrix<SG_T> getMatrixUsingMemcpy
			(cv::Mat, int, int);

		template<typename SG_T> static SGMatrix<SG_T> getMatrixUsingConstructor
			(cv::Mat, int, int);
};

template<typename SG_T> CDenseFeatures<SG_T>* CV2SGFactory::getDensefeatures
	(cv::Mat cvMat, CV2SGOptions option)
{
	SGMatrix<SG_T> sgMat=CV2SGFactory::getSGMatrix<SG_T>(cvMat, option);
	CDenseFeatures<SG_T>* features=new CDenseFeatures<SG_T>(sgMat);
	return features;
}

template<typename SG_T> SGMatrix<SG_T> CV2SGFactory::getSGMatrix
	(cv::Mat cvMat, CV2SGOptions option)
{
	SGMatrix<SG_T> sgMat;
	int num_rows=cvMat.rows;
	int num_cols=cvMat.cols;
	const int inType=OpenCVTypeName<SG_T>::get_opencv_type();
	cvMat.convertTo(cvMat,inType);
	switch (option)
	{
		case CV2SG_CONSTRUCTOR:
		sgMat=CV2SGFactory::getMatrixUsingConstructor<SG_T>
			(cvMat, num_rows, num_cols);
		break;

		case CV2SG_MANUAL:
		sgMat=CV2SGFactory::getMatrixUsingManual<SG_T>
			(cvMat, num_rows, num_cols);
		break;

		case CV2SG_MEMCPY:
		sgMat=CV2SGFactory::getMatrixUsingMemcpy<SG_T>
			(cvMat, num_rows, num_cols);
		break;
	} 
return sgMat;
}

template<typename SG_T> SGMatrix<SG_T> CV2SGFactory::getMatrixUsingManual
	(cv::Mat cvMat, int num_rows, int num_cols)
{
	SGMatrix<SG_T> sgMat(num_rows, num_cols);
	for(int i=0; i<num_rows; i++)
	{
		for(int j=0; j<num_cols; j++)
		{
			sgMat(i,j)=cvMat.at<SG_T>(i, j);
		}
	}
	return sgMat;
}

template<typename SG_T> SGMatrix<SG_T> CV2SGFactory::getMatrixUsingMemcpy
	(cv::Mat cvMat, int num_rows, int num_cols)
{
	SGMatrix<SG_T> sgMat(num_rows, num_cols);
	memcpy(sgMat.matrix, cvMat.data, num_rows*num_cols*sizeof(SG_T));
	SGMatrix<SG_T>::transpose_matrix(sgMat.matrix, num_cols, num_rows);
	return sgMat;
}

template<typename SG_T> SGMatrix<SG_T> CV2SGFactory::getMatrixUsingConstructor
	(cv::Mat cvMat, int num_rows, int num_cols)
{
	cvMat=cvMat.t();
	SGMatrix<SG_T> sgMat((SG_T*)cvMat.data, num_rows, num_cols, false);
	return sgMat;
}
}
#endif /*CV2_SGMATRIX_FACTORY_H_*/
#endif /* HAVE_OPENCV */
