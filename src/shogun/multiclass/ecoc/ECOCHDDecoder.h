/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/ecoc/ECOCDecoder.h>
#include <shogun/multiclass/ecoc/ECOCUtil.h>

namespace shogun
{

/** Hamming Distance Decoder */
class CECOCHDDecoder: public CECOCDecoder
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
