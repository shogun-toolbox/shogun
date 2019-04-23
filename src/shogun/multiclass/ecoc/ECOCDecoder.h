/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Yuyu Zhang
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
class ECOCDecoder: public SGObject
{
public:
    /** constructor */
    ECOCDecoder() {}

    /** destructor */
    ~ECOCDecoder() {}

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
