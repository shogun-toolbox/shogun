#ifndef LDA_H_
#define LDA_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/eigen3.h>

namespace shogun
{
class CFisherLDA: public CDimensionReductionPreprocessor
{
    public:

        /** standard constructor
         * @param number of dimensions to retain
         */
        CFisherLDA(int32_t num_dim=0);
        
        /** destructor */
        virtual ~CFisherLDA();

        /** initialize preprocessor from features and corresponding labels
         * @param features
         * @param labels
         */
        virtual bool init(CFeatures* features, CLabels* labels);

        /** cleanup */
		virtual void cleanup();

        /** apply preprocessor to feature matrix
		 * @param features features
		 * @return processed feature matrix
		 */
        virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);
         
        /** apply preprocessor to feature matrix
		 * @param features features
		 * @return processed feature vector
		 */
        virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);
    
        /** get transformation matrix aka the required number of eigenvectors
        */
        SGMatrix<float64_t> get_transformation_matrix();

        /** get eigenvalues of LDA
        */
        SGVector<float64_t> get_eigenvalues();
    
        /** get mean vector of the original data
        */
        SGVector<float64_t> get_mean();

		/** @return object name */
		virtual const char* get_name() const { return "FisherLDA"; }

		/** @return a type of preprocessor */
		//virtual EPreprocessorType get_type() const { return P_FisherLDA; }
 
    protected:

        void init();

		/** transformation matrix */
		SGMatrix<float64_t> m_transformation_matrix;
		/** num dim */
		int32_t num_dim;
		/** num old dim */
		int32_t num_old_dim;
		/** mean vector */
		SGVector<float64_t> m_mean_vector;
		/** eigenvalues vector */
		SGVector<float64_t> m_eigenvalues_vector;

}; 
}
#endif //HAVE_EIGEN3
#endif //ifndef

