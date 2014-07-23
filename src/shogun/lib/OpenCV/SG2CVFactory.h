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

#ifndef SG2_CVMAT_FACTORY_H_
#define SG2_CVMAT_FACTORY_H_

#include <opencv2/core/core.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/OpenCV/OpenCVTypeName.h>

namespace shogun 
{
/** @brief CV2SGFactory converts Shogun's SGMatrix and DenseFeatures into 
 * OpenCV's cv::Mat format. The data is directly copied to the memory block
 * pointed by the cv::Mat.
 */
class SG2CVFactory
{
	public:
		/* constructor */
		SG2CVFactory();
		/* destructor */
		~SG2CVFactory();
		/* get cv::Mat from SGMatrix. 
		 * @param SGMatrix which is to be converted
		 * @param specify the OpenCV data type. i.e (CV_8U, CV_8S, CV_16U, 
		 * CV_16S, CV_32S, CV_32F, CV_64F)
		 * @return Mat object of the specified data type
		 */
		template <typename SG_T> static cv::Mat get_cvMat(SGMatrix<SG_T> sgMat,
				int cv_type);
		/* get cv::Mat from Densefeatures
		 * @param DenseFeatures pointer 
		 * @param specify the OpenCV data type. i.e (CV_8U, CV_8S, CV_16U,
		 * CV_16S, CV_32S, CV_32F, CV_64F)
		 * @return Mat object of the specified data type
		 */
		template <typename SG_T> static cv::Mat get_cvMat_from_features
			(CDenseFeatures<SG_T>* sgDense, int cv_type);

	private: 
		template <typename SG_T, typename CV_T> static cv::Mat get_cvMat
			(SGMatrix<SG_T> sgMat);
};

template <typename SG_T, typename CV_T> cv::Mat SG2CVFactory::get_cvMat
	(SGMatrix<SG_T> sgMat)
{
	int num_rows=sgMat.num_rows;
	int num_cols=sgMat.num_cols;
	const int outType=OpenCVTypeName<CV_T>::get_opencv_type();
	const int inType=OpenCVTypeName<SG_T>::get_opencv_type(); 
	cv::Mat cvMat(num_cols, num_rows, inType);
	memcpy((SG_T*)cvMat.data, sgMat.matrix, num_rows*num_cols*sizeof(SG_T));
	cvMat.convertTo(cvMat,outType);
	return cvMat.t();
}

template <typename SG_T> cv::Mat SG2CVFactory::get_cvMat
	(SGMatrix<SG_T> sgMat, int cv_type)
{
	cv::Mat cvMat;
	switch(cv_type)
	{
		case CV_8U:
			cvMat=SG2CVFactory::get_cvMat<SG_T, unsigned char>(sgMat);
			break;

		case CV_8S:
			cvMat=SG2CVFactory::get_cvMat<SG_T, signed char>(sgMat);
			break;

		case CV_16U:
			cvMat=SG2CVFactory::get_cvMat<SG_T, unsigned short>(sgMat);
			break;

		case CV_16S:
			cvMat=SG2CVFactory::get_cvMat<SG_T, signed short>(sgMat);
			break;

		case CV_32S:
			cvMat=SG2CVFactory::get_cvMat<SG_T, int>(sgMat);
			break;

		case CV_32F:
			cvMat=SG2CVFactory::get_cvMat<SG_T, float>(sgMat);
			break;

		case CV_64F:
			cvMat=SG2CVFactory::get_cvMat<SG_T, double>(sgMat);
			break;
	}
	return cvMat;
}

template <typename SG_T> cv::Mat SG2CVFactory::get_cvMat_from_features
	(CDenseFeatures<SG_T>* sgDense, int cv_type)
{
	SGMatrix<SG_T> sgMat=sgDense->get_feature_matrix();
	cv::Mat cvMat=SG2CVFactory::get_cvMat<SG_T>(sgMat, cv_type);
	return cvMat;
}
}
#endif /*SG2_CVMAT_FACTORY_H_*/
#endif /* HAVE_OPENCV */
