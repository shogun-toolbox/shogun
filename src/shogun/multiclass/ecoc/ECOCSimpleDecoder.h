/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCSIMPLEDECODER_H__
#define ECOCSIMPLEDECODER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCDecoder.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** A decoder that computes some simple distances between
 * the binary classification results and the codes to select
 * the class with the smallest distance.
 */
class CECOCSimpleDecoder: public CECOCDecoder
{
public:
    /** constructor */
    CECOCSimpleDecoder() {}

    /** destructor */
    virtual ~CECOCSimpleDecoder() {}

    /** get name */
    virtual const char* get_name() const { return "ECOCSimpleDecoder"; }

    /** decide label.
     * @param outputs outputs by classifiers
     * @param codebook ECOC codebook
     */
    virtual int32_t decide_label(const SGVector<float64_t> outputs, const SGMatrix<int32_t> codebook);

protected:
    /** whether to turn the output into binary before decoding */
    virtual bool binary_decoding()=0;

    /** compute distance */
    virtual float64_t compute_distance(SGVector<float64_t> outputs, const int32_t *code)=0;
};

} /* shogun */

#endif /* end of include guard: ECOCSIMPLEDECODER_H__ */

