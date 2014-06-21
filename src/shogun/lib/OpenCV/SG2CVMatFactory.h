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

namespace shogun {

enum SG2CVOptions {SG2CV_CONSTRUCTOR, SG2CV_MANUAL, SG2CV_MEMCPY};


class SG2CVMatFactory
{
public:
  SG2CVMatFactory();
  ~SG2CVMatFactory();
  template <typename SG_T> static cv::Mat getcvMat(SGMatrix<SG_T> sgMat, int cv_type, SG2CVOptions=SG2CV_MEMCPY);
  template <typename SG_T, typename CV_T> static cv::Mat getcvMat(SGMatrix<SG_T> sgMat, SG2CVOptions=SG2CV_MEMCPY);

private:
  template<typename SG_T, typename CV_T> static cv::Mat getMatUsingManual(SGMatrix<SG_T>, int, int, int, int);
  template<typename SG_T, typename CV_T> static cv::Mat getMatUsingMemcpy(SGMatrix<SG_T>, int, int, int, int);
  template<typename SG_T, typename CV_T> static cv::Mat getMatUsingConstructor(SGMatrix<SG_T>, int, int, int, int);
};

template <typename SG_T> cv::Mat SG2CVMatFactory::getcvMat(SGMatrix<SG_T> sgMat, int cv_type, SG2CVOptions option)
{
  cv::Mat cvMat;

  switch(cv_type)
  {
    case CV_8U:
      cvMat = SG2CVMatFactory::getcvMat<SG_T, unsigned char>(sgMat, option);
      break;

    case CV_8S:
      cvMat = SG2CVMatFactory::getcvMat<SG_T, signed char>(sgMat, option);
      break;

    case CV_16U:
      cvMat = SG2CVMatFactory::getcvMat<SG_T, unsigned short>(sgMat, option);
      break;

    case CV_16S:
      cvMat = SG2CVMatFactory::getcvMat<SG_T, signed short>(sgMat, option);
      break;

    case CV_32S:
      cvMat = SG2CVMatFactory::getcvMat<SG_T, int>(sgMat, option);
      break;

    case CV_32F:
      cvMat = SG2CVMatFactory::getcvMat<SG_T, float>(sgMat, option);
      break;

    case CV_64F:
      cvMat = SG2CVMatFactory::getcvMat<SG_T, double>(sgMat, option);
      break;
  }

  return cvMat;
}

template <typename SG_T, typename CV_T> cv::Mat SG2CVMatFactory::getcvMat(SGMatrix<SG_T> sgMat, SG2CVOptions option)
{
  cv::Mat cvMat;

  int num_rows = sgMat.num_rows;
  int num_cols = sgMat.num_cols;

  const int myType = OpenCVTypeName<CV_T>::get_opencv_type();
  const int initType = OpenCVTypeName<SG_T>::get_opencv_type();

  switch(option)
  {
    case SG2CV_MEMCPY:
      cvMat = SG2CVMatFactory::getMatUsingMemcpy<SG_T, CV_T>(sgMat, num_rows, num_cols, myType, initType);
      break;

    case SG2CV_MANUAL:
      cvMat = SG2CVMatFactory::getMatUsingManual<SG_T, CV_T>(sgMat, num_rows, num_cols, myType, initType);
      break;

    case SG2CV_CONSTRUCTOR:
      cvMat = SG2CVMatFactory::getMatUsingConstructor<SG_T, CV_T>(sgMat, num_rows, num_cols, myType, initType);
      break;
  }

  return cvMat;
}

template<typename SG_T, typename CV_T> cv::Mat SG2CVMatFactory::getMatUsingManual(SGMatrix<SG_T> sgMat, int num_rows, int num_cols, int myType, int initType)
{
  cv::Mat cvMat(num_rows, num_cols, myType);
  for(int i=0; i<num_rows; i++)
  {
    for(int j=0; j<num_cols; j++)
    {
      cvMat.at<CV_T>(i, j)=sgMat(i, j);
    }
  }
  return cvMat;
}

template<typename SG_T, typename CV_T> cv::Mat SG2CVMatFactory::getMatUsingMemcpy(SGMatrix<SG_T> sgMat, int num_rows, int num_cols, int myType, int initType)
{
  cv::Mat cvMat(num_rows, num_cols, initType);
  memcpy((SG_T*)cvMat.data, sgMat.matrix, num_rows*num_cols*sizeof(SG_T));
  cvMat.convertTo(cvMat,myType);
  return cvMat.t();
}

template<typename SG_T, typename CV_T> cv::Mat SG2CVMatFactory::getMatUsingConstructor(SGMatrix<SG_T> sgMat, int num_rows, int num_cols, int myType, int initType)
{
  cv::Mat cvMat(num_rows, num_cols, initType, (void*)sgMat.matrix);
  cvMat.convertTo(cvMat, myType);
  return cvMat.t();
}

}

#endif /*SG2_CVMAT_FACTORY_H_*/
#endif /* HAVE_OPENCV */
