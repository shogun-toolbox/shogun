
 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
 
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
 
#include <iostream>
 
using namespace cv;
using namespace shogun;
 
int main()
{
	init_shogun_with_defaults();
		
	Mat cvMat = Mat::eye(3,3,CV_8UC1);
	cvMat.at<unsigned char>(0,1) = 3;
	SGMatrix<unsigned char> sgMat(cvMat.data,3,3,false);
	
	std::cout << cvMat << std::endl ;
	sgMat.display_matrix();
	
	
	return 0;
}
	
