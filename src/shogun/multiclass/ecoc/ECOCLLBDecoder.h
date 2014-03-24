/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
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
class CECOCLLBDecoder: public CECOCSimpleDecoder
{
public:
    /** constructor */
    CECOCLLBDecoder() {}

    /** destructor */
    virtual ~CECOCLLBDecoder() {}

    /** get name */
    virtual const char* get_name() const { return "ECOCLLBDecoder"; }

protected:
    /** whether to turn the output into binary before decoding */
    virtual bool binary_decoding() { return false; }

    /** compute distance */
    virtual float64_t compute_distance(SGVector<float64_t> outputs, const int32_t *code);
};

} /* shogun */

#endif /* end of include guard: ECOCLLBDECODER_H__ */

