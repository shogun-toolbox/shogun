#ifndef _STREAMING_SIMPLEFEATURES__H__
#define _STREAMING_SIMPLEFEATURES__H__

#include "lib/common.h"
#include "lib/Time.h"
#include "lib/Mathematics.h"
#include "features/Features.h"
#include "lib/InputParser.h"

namespace shogun
{
/** @brief Features that support streaming dot products among other operations.
 *
 * DotFeatures support the following operations:
 *
 * - a way to obtain the dimensionality of the feature space, i.e. \f$\mbox{dim}({\cal X})\f$
 *
 * - dot product between feature vectors:
 *
 *   \f[r = {\bf x} \cdot {\bf x'}\f]
 *
 * - dot product between feature vector and a dense vector \f${\bf z}\f$:
 *
 *   \f[r = {\bf x} \cdot {\bf z}\f]
 *
 * - multiplication with a scalar \f$\alpha\f$ and addition to a dense vector \f${\bf z}\f$:
 *
 *   \f[ {\bf z'} = \alpha {\bf x} + {\bf z} \f]
 *
 * - iteration over all (potentially) non-zero features of \f${\bf x}\f$
 * 
 */
template <class T>
class CStreamingDotFeatures : public CFeatures
{

public:

	CStreamingDotFeatures();

	CStreamingDotFeatures(CStreamingFile* file,
			      bool is_labelled,
			      int32_t size);
		
	~CStreamingDotFeatures();
		
	void init();

	void init(CStreamingFile *file, bool is_labelled, int32_t size);

	void start_parser();

	void end_parser();

	int32_t get_next_example();

	void get_vector(SGVector<T> &vec);

	void get_label(float64_t &label);

	void release_example();

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space();

	virtual float64_t dot(SGVector<T> &vec);
		
	virtual float64_t dense_dot(SGVector<T> &vec);

	virtual void add_to_dense_vec(float64_t alpha, SGVector<T> &vec, bool abs_val=false);

	int32_t get_num_features();
		
	EFeatureClass get_feature_class();
		
protected:
		
	/// feature weighting in combined dot features
	float64_t combined_weight;

	CInputParser<T> parser;
	CStreamingFile* working_file;
	T* current_vector;
	float64_t current_label;
	int32_t current_length;
	bool has_labels;
};
	
	
template <class T>
void CStreamingDotFeatures<T>::init()
{
	working_file=NULL;
	current_vector=NULL;
	current_length=-1;
}

template <class T>
void CStreamingDotFeatures<T>::init(CStreamingFile* file,
				    bool is_labelled,
				    int32_t size)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	parser.init(file, is_labelled, size);
}
	

template <class T>
CStreamingDotFeatures<T>::CStreamingDotFeatures()
{
	init();
}

template <class T>
CStreamingDotFeatures<T>::CStreamingDotFeatures(CStreamingFile* file,
						bool is_labelled,
						int32_t size)
{
	init(file, is_labelled, size);
}

template <class T>
CStreamingDotFeatures<T>::~CStreamingDotFeatures()
{
	parser.end_parser();
}

template <class T>
void CStreamingDotFeatures<T>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template <class T>
void CStreamingDotFeatures<T>::end_parser()
{
	parser.end_parser();
}

template <class T>
int32_t CStreamingDotFeatures<T>::get_next_example()
{
	int32_t ret_value;
	ret_value = parser.get_next_example(current_vector,
					    current_length,
					    current_label);

	return ret_value;
}

template <class T>
void CStreamingDotFeatures<T>::get_vector(SGVector<T> &vec)
{
	vec.vector=current_vector;
	vec.length=current_length;
}

template <class T>
void CStreamingDotFeatures<T>::get_label(float64_t &label)
{
	ASSERT(has_labels);

	label=current_label;
}
	
template <class T>
void CStreamingDotFeatures<T>::release_example()
{
	parser.finalize_example();
}

template <class T>
int32_t CStreamingDotFeatures<T>::get_dim_feature_space()
{
	return current_length;
}

template <class T>
float64_t CStreamingDotFeatures<T>::dot(SGVector<T> &sgvec1)
{
	int32_t len1;
	len1=sgvec1.length;
				
	if (len1 != current_length)
		SG_ERROR("Lengths %d and %d not equal while computing dot product!\n", len1, current_length);

	float64_t result=CMath::dot(current_vector, sgvec1.vector, len1);
	return result;
}

template <class T>
float64_t CStreamingDotFeatures<T>::dense_dot(SGVector<T> &sgvec1)
{
	int32_t len1=sgvec1.length;

	ASSERT(len1==current_length);
	float64_t result=0;
		
	for (int32_t i=0; i<current_length; i++)
		result+=current_vector[i]*sgvec1.vector[i];

	return result;
}

template <class T>
void CStreamingDotFeatures<T>::add_to_dense_vec(float64_t alpha,
						SGVector<T> &vec,
						bool abs_val)
{
	ASSERT(vec.length==current_length);

	if (abs_val)
	{
		for (int32_t i=0; i<current_length; i++)
			vec.vector[i]+=alpha*CMath::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_length; i++)
			vec.vector[i]+=alpha*current_vector[i];
	}
}

template <class T>
int32_t CStreamingDotFeatures<T>::get_num_features()
{
	return current_length;
}

template <class T>
EFeatureClass CStreamingDotFeatures<T>::get_feature_class()
{
	return C_SIMPLE;
}

}
#endif // _STREAMING_SIMPLEFEATURES__H__
