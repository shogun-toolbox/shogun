/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang, Bjoern Esser
 */

#ifndef ECOCLLBDECODER_H__
#define ECOCLLBDECODER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCSimpleDecoder.h>

namespace shogun
{

/** Margin Loss based decoder.
 * Using OVREncoder with this Decoder should be equivlalent to
 * traditional OVR Strategy.
 */
class ECOCLLBDecoder: public ECOCSimpleDecoder
{
public:
    /** constructor */
    ECOCLLBDecoder() {}

    /** destructor */
    ~ECOCLLBDecoder() override {}

    /** get name */
    const char* get_name() const override { return "ECOCLLBDecoder"; }

protected:
    /** whether to turn the output into binary before decoding */
    bool binary_decoding() override { return false; }

    /** compute distance */
    float64_t compute_distance(SGVector<float64_t> outputs, const int32_t *code) override;
};

} /* shogun */

#endif /* end of include guard: ECOCLLBDECODER_H__ */

