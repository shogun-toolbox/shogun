#ifdef HAVE_OPENCV

#include <opencv2/highgui/highgui.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/OpenCV/CV2SGMatrixFactory.h>

namespace shogun{

class CV2FeaturesFactory
{
	public:
	
	CV2FeaturesFactory();
	
	virtual ~CV2FeaturesFactory();
			
	template <typename T> static CDenseFeatures<T>* getDenseFeatures(cv::Mat, CV2SGOptions=CV2SG_CONSTRUCTOR);
};

template<typename T> CDenseFeatures<T>* CV2FeaturesFactory::getDenseFeatures(cv::Mat cvMat, CV2SGOptions option)
{
	SGMatrix<T> sgMat = CV2SGMatrixFactory::getMatrix<T>(cvMat, option);
	CDenseFeatures<T>* features = new CDenseFeatures<T>(sgMat);
	return features;
}

}

#endif /* HAVE_OPENCV */
