/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCAEDDECODER_H__
#define ECOCAEDDECODER_H__


#include <multiclass/ecoc/ECOCSimpleDecoder.h>
#include <mathematics/Math.h>

namespace shogun
{

/** Attenuated Euclidean Distance Decoder.
 *
 * \f[
 * AED(q, b_i) = \sqrt{\sum_{j=1}^n (q^j-b_i^j)^2 |b_i^j|}
 * \f]
 */
class CECOCAEDDecoder: public CECOCSimpleDecoder
{
public:
    /** constructor */
    CECOCAEDDecoder() {}

    /** destructor */
    virtual ~CECOCAEDDecoder() {}

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
            dist += (outputs[i]-code[i])*(outputs[i]-code[i]) * CMath::abs(code[i]);
        return CMath::sqrt(dist);
    }
};

} /* shogun */


#endif /* end of include guard: ECOCAEDDECODER_H__ */
