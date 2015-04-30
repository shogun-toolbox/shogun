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

#ifndef CV2_SGMATRIX_FACTORY_H_
#define CV2_SGMATRIX_FACTORY_H_

#include <opencv2/core/core.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/OpenCV/OpenCVTypeName.h>

namespace shogun
{
/** @brief CV2SGFactory copies data from OpenCV's cv::Mat data structure into
 * Shogun's SGMatrix and DenseFeatures format. The Mat is the basic image
 * container class of OpenCV which is also a general matrix class. For this
 * conversion, you need to have 2-D matrices.
 */
class CV2SGFactory
{
	public:
		/*constructor*/
		CV2SGFactory();
		/*destructor*/
		~CV2SGFactory();
		/* get SGMatrix from the Mat object. The choice of the data type for the
		 * resulting SGMatrix is completely yours irrespective of the cv::Mat
		 * data type
		 * @param cv::Mat to be converted
		 * @return SGMatrix of the specified data type
		 */
		template<typename SG_T> static SGMatrix<SG_T> get_sgmatrix(cv::Mat);
		/* get DenseFeatures from the Mat object. The choice of the data type
		 * for the resulting DenseFeatures is completely yours irrespective of
		 * the cv::Mat data type.
		 * @param cv::Mat to be converted
		 * @return DenseFeatures pointer of the specified data type
		 */
		template<typename SG_T> static CDenseFeatures<SG_T>* get_dense_features(cv::Mat);
};

template<typename SG_T> CDenseFeatures<SG_T>* CV2SGFactory::get_dense_features
	(cv::Mat cvMat)
{
	SGMatrix<SG_T> sgMat=CV2SGFactory::get_sgmatrix<SG_T>(cvMat);
	CDenseFeatures<SG_T>* features=new CDenseFeatures<SG_T>(sgMat);
	return features;
}

template<typename SG_T> SGMatrix<SG_T> CV2SGFactory::get_sgmatrix
	(cv::Mat cvMat)
{
	int num_rows=cvMat.rows;
	int num_cols=cvMat.cols;
	SGMatrix<SG_T> sgMat(num_rows, num_cols);
	const int inType=OpenCVTypeName<SG_T>::get_opencv_type();
	cvMat.convertTo(cvMat,inType);
	memcpy(sgMat.matrix, cvMat.data, num_rows*num_cols*sizeof(SG_T));
	SGMatrix<SG_T>::transpose_matrix(sgMat.matrix, num_cols, num_rows);
	return sgMat;
}
}
#endif /*CV2_SGMATRIX_FACTORY_H_*/
#endif /* HAVE_OPENCV */
