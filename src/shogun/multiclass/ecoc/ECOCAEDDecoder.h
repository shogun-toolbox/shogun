/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang, Bjoern Esser
 */

#ifndef ECOCAEDDECODER_H__
#define ECOCAEDDECODER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCSimpleDecoder.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** Attenuated Euclidean Distance Decoder.
 *
 * \f[
 * AED(q, b_i) = \sqrt{\sum_{j=1}^n (q^j-b_i^j)^2 |b_i^j|}
 * \f]
 */
class ECOCAEDDecoder: public ECOCSimpleDecoder
{
public:
    /** constructor */
    ECOCAEDDecoder() {}

    /** destructor */
    virtual ~ECOCAEDDecoder() {}

    /** get name */
    virtual const char* get_name() const { return "ECOCAEDDecoder"; }


protected:
    /** whether to turn the output into binary before decoding */
    virtual bool binary_decoding()
    {
        return false;
    }

    /** compute distance */
    virtual float64_t compute_distance(SGVector<float64_t> outputs, const int32_t *code)
    {
        float64_t dist = 0;
        for (int32_t i=0; i < outputs.vlen; ++i)
            dist += (outputs[i]-code[i])*(outputs[i]-code[i]) * Math::abs(code[i]);
		return std::sqrt(dist);
	}
};

} /* shogun */


#endif /* end of include guard: ECOCAEDDECODER_H__ */
