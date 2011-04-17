
#ifndef _STREAMINGFEATURES__H__
#define _STREAMINGFEATURES__H__

#include "lib/common.h"
#include "features/Features.h"
#include "lib/File.h"
#include "lib/parser.h"
#include <pthread.h>

namespace shogun
{
	//class CFeatures;

	class CStreamingFeatures : public CFeatures
	{
		void init(void);

	public:

		CStreamingFeatures();

		CStreamingFeatures(FILE* file, int32_t buffer_size);

		CStreamingFeatures(const CStreamingFeatures & orig);

		CStreamingFeatures(CFile* loader);

		virtual ~CStreamingFeatures() {	}

		inline virtual const char* get_name() const { return "StreamingFeatures"; }

		inline EFeatureType get_feature_type()
		{
			return F_DREAL;
		}

		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }

		virtual inline int32_t	get_num_vectors() { return -1; }

		virtual int32_t get_size() { return sizeof(*this); }

		virtual CFeatures* duplicate() const
		{
			return NULL;
		}


		void start_parser();

		void end_parser();

		virtual int32_t get_dim_feature_space()
		{
			return current_length;
		}

		virtual int32_t get_next_feature_vector(float64_t* &feature_vector, int32_t &length, int32_t &label);


	protected:

		input_parser parser;

		FILE* working_file;

		float64_t* current_feature_vector;
		int32_t current_label;
		int32_t current_length;

	};
}
#endif	/* _STREAMINGFEATURES__H__ */
