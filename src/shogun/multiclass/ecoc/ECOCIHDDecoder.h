/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCIHDDECODER_H__
#define ECOCIHDDECODER_H__
#ifdef HAVE_LAPACK

#include <multiclass/ecoc/ECOCDecoder.h>

namespace shogun
{

/** Inverse Hamming Decoding.
 *
 * \f[
 * IHD(q, B) = \arg\max (\Delta^{-1}D)
 * \f]
 * where
 * \f[
 * \Delta_{ij} = HD(b_i,b_j)
 * \f]
 * and
 * \f[
 * D_i = HD(q, b_i)
 * \f]
 */
class CECOCIHDDecoder: public CECOCDecoder
{
public:
    /** constructor */
    CECOCIHDDecoder() {}

    /** destructor */
    virtual ~CECOCIHDDecoder() {}

    /** get name */
    virtual const char* get_name() const { return "ECOCIHDDecoder"; }

    /** decide label.
     * @param outputs outputs by classifiers
     * @param codebook ECOC codebook
     */
    virtual int32_t decide_label(const SGVector<float64_t> outputs, const SGMatrix<int32_t> codebook);

protected:
    /** update delta cache */
    void update_delta_cache(const SGMatrix<int32_t> codebook);

    SGMatrix<float64_t> m_delta;
    SGMatrix<int32_t> m_codebook;
};

} // namespace shogun

#endif // HAVE_LAPACK
#endif /* end of include guard: ECOCIHDDECODER_H__ */

