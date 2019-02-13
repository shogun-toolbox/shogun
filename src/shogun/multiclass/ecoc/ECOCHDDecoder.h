/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef ECOCHDDECODER_H__
#define ECOCHDDECODER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCSimpleDecoder.h>
#include <shogun/multiclass/ecoc/ECOCUtil.h>

namespace shogun
{

/** Hamming Distance Decoder */
class SHOGUN_EXPORT CECOCHDDecoder: public CECOCSimpleDecoder
{
public:
    /** constructor */
    CECOCHDDecoder() {}

    /** destructor */
    virtual ~CECOCHDDecoder() {}

    /** get name */
    virtual const char* get_name() const
    {
        return "ECOCHDDecoder";
    }

protected:
    /** whether to turn the output into binary before decoding */
    virtual bool binary_decoding()
    {
        return true;
    }

    /** compute distance */
    virtual float64_t compute_distance(SGVector<float64_t> outputs, const int32_t *code)
    {
        return CECOCUtil::hamming_distance(outputs.vector, code, outputs.vlen);
    }
};

}

#endif /* end of include guard: ECOCHDDECODER_H__ */
