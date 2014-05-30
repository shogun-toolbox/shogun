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

namespace shogun{

enum SG2CVOptions {SG2CV_CONSTRUCTOR, SG2CV_MANUAL, SG2CV_MEMCPY};

template <typename T> struct SG2CVTypeName;

template<> struct SG2CVTypeName<uint8_t>
{
	static const int get()
	{
		return 0;
	}
};

template<> struct SG2CVTypeName<int8_t>
{
	static const int get()
	{
		return 1;
	}
};

template<> struct SG2CVTypeName<uint16_t>
{
	static const int get()
	{
		return 2;
	}
};

template<> struct SG2CVTypeName<int16_t>
{
	static const int get()
	{
		return 3;
	}
};

template<> struct SG2CVTypeName<int32_t>
{
	static const int get()
	{
		return 4;
	}
};

template<> struct SG2CVTypeName<float32_t>
{
	static const int get()
	{
		return 5;
	}
};

template<> struct SG2CVTypeName<float64_t>
{
    static const int get()
    {
    	return 6;
    }
};

class SG2CVMatFactory
{
public:
	
	SG2CVMatFactory();	
	
	~SG2CVMatFactory();
	
	template <typename T> static cv::Mat getMatrix(CDenseFeatures<float64_t>* A,
			SG2CVOptions=SG2CV_CONSTRUCTOR);
};

template <typename T> cv::Mat SG2CVMatFactory::getMatrix(CDenseFeatures<float64_t>* A,
		SG2CVOptions option)
{
	SGMatrix<float64_t> sgMat=((CDenseFeatures<float64_t>*)A)->get_feature_matrix();
	int num_rows=sgMat.num_rows;
	int num_cols=sgMat.num_cols;
	const int myType=SG2CVTypeName<T>::get();

	switch(option)
	{
		case SG2CV_CONSTRUCTOR:
		{
			cv::Mat cvMat(num_rows, num_cols, CV_64FC1, (void*)sgMat.matrix);
			cvMat.convertTo(cvMat, myType);
			cvMat.t();
			return cvMat;
		}
		
		case SG2CV_MANUAL:
		{
			cv::Mat cvMat(num_rows, num_cols, CV_64FC1);
			for(int i=0; i<num_rows; i++)
			{
				for(int j=0; j<num_cols; j++)
				{
					cvMat.at<double>(i, j)=sgMat(i, j);
				}
			}
			cvMat.convertTo(cvMat, myType);
			return cvMat;
		}

		case SG2CV_MEMCPY:
		{
			cv::Mat cvMat(num_rows, num_cols, CV_64FC1);
			memcpy((double*)cvMat.data, sgMat.matrix, num_rows*num_cols*sizeof(double));
			cvMat.convertTo(cvMat,myType);
			cvMat.t();
			return cvMat;
		}
		
		default:
		{
			cv::Mat cvMat(num_rows, num_cols, CV_64FC1);
			memcpy((double*)cvMat.data, sgMat.matrix, num_rows*num_cols*sizeof(double));
			cvMat.convertTo(cvMat, myType);
			cvMat.t();
			return cvMat;
		}
	}
}

}

#endif /*SG2_CVMAT_FACTORY_H_*/
#endif /* HAVE_OPENCV */
