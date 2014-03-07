#ifndef _WHITENINGPREPROCESSOR_H
#define _WHITENINGPREPROCESSOR_H

#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>



namespace shogun
{
	//template<class ST> class CWhiteningPreprocessor;
	/*
	performs a whitening transform on the input matrix such that 
	it has an identity co variance matrix
	
	*/

	class CWhiteningPreprocessor : public CDensePreprocessor<float64_t>
	{
		public:
			CWhiteningPreprocessor();
			virtual ~CWhiteningPreprocessor();

			virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

			virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

			virtual bool init(CFeatures* features);

				/// cleanup
		virtual void cleanup();
		/// initialize preprocessor from file
		virtual bool load(FILE* f);
		/// save preprocessor init-data to file
		virtual bool save(FILE* f);
/** @return object name */
		virtual const char* get_name() const { return "whitening preprocessor"; }

	
	};


}

#endif