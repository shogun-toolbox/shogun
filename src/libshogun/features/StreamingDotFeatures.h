#ifndef _STREAMING_DOTFEATURES__H__
#define _STREAMING_DOTFEATURES__H__

#include "lib/common.h"
#include "lib/Time.h"
#include "lib/Mathematics.h"
#include "features/Features.h"
#include "lib/StreamingFile.h"

namespace shogun
{
/** @brief Streaming features that support dot products among other operations.
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

class CStreamingDotFeatures : public CFeatures
{

public:

	CStreamingDotFeatures();

	CStreamingDotFeatures(CStreamingFile* file,
			      bool is_labelled,
			      int32_t size);
		
	~CStreamingDotFeatures();
		
	virtual void init();

	virtual void init(CStreamingFile *file, bool is_labelled, int32_t size);

	virtual void start_parser();

	virtual void end_parser();

	virtual int32_t get_next_example();

	virtual void release_example();

	virtual float64_t dot(CStreamingDotFeatures* df)=0;
	
	virtual float64_t dense_dot(const SGVector<float64_t> &vec)=0;

	virtual void add_to_dense_vec(float64_t alpha, const SGVector<float64_t> &vec, bool abs_val=false)=0;
	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space();

	virtual int32_t get_num_features();
		
protected:
		
	/// feature weighting in combined dot features
	float64_t combined_weight;
};
}
#endif // _STREAMING_DOTFEATURES__H__
