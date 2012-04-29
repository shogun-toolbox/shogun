/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCDECODER_H__
#define ECOCDECODER_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>

namespace shogun
{

class CECOCDecoder: public CSGObject
{
public:
    /** constructor */
    CECOCDecoder() {}

    /** destructor */
    ~CECOCDecoder() {}

    /** get name */
    const char* get_name() const
    {
        return "ECOCDecoder";
    }


    /** decide label.
     * @param outputs outputs by classifiers
     * @param codebook ECOC codebook
     */
    virtual int32_t decide_label(const SGVector<float64_t> &outputs, const SGMatrix<int32_t> &codebook);

protected:
    /** whether to turn the output into binary before decoding */
    virtual bool binary_decoding()=0;

    /** compute distance */
    virtual float64_t compute_distance(const SGVector<float64_t> &outputs, const int32_t *code)=0;
};

}

#endif /* end of include guard: ECOCDECODER_H__ */
