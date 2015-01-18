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

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

/** An ECOC decoder describe how to decode the
 * classification results of the binary classifiers
 * into a multiclass label according to the ECOC
 * codebook.
 */
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
    virtual int32_t decide_label(const SGVector<float64_t> outputs, const SGMatrix<int32_t> codebook)=0;

protected:
    /** turn 2-class labels into binary */
    SGVector<float64_t> binarize(const SGVector<float64_t> query);
};

}

#endif /* end of include guard: ECOCDECODER_H__ */
