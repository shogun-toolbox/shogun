/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang
 */

#ifndef ECOCIHDDECODER_H__
#define ECOCIHDDECODER_H__
#ifdef HAVE_LAPACK

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCDecoder.h>

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
class ECOCIHDDecoder: public ECOCDecoder
{
public:
    /** constructor */
    ECOCIHDDecoder() {}

    /** destructor */
    ~ECOCIHDDecoder() override {}

    /** get name */
    const char* get_name() const override { return "ECOCIHDDecoder"; }

    /** decide label.
     * @param outputs outputs by classifiers
     * @param codebook ECOC codebook
     */
    int32_t decide_label(const SGVector<float64_t> outputs, const SGMatrix<int32_t> codebook) override;

protected:
    /** update delta cache */
    void update_delta_cache(const SGMatrix<int32_t> codebook);

    SGMatrix<float64_t> m_delta;
    SGMatrix<int32_t> m_codebook;
};

} // namespace shogun

#endif // HAVE_LAPACK
#endif /* end of include guard: ECOCIHDDECODER_H__ */

