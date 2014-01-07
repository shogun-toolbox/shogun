/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <lib/config.h>

#ifdef HAVE_LAPACK

#include <multiclass/ecoc/ECOCIHDDecoder.h>
#include <multiclass/ecoc/ECOCUtil.h>
#include <mathematics/Math.h>
#include <mathematics/lapack.h>

using namespace shogun;


int32_t CECOCIHDDecoder::decide_label(const SGVector<float64_t> outputs, const SGMatrix<int32_t> codebook)
{
    update_delta_cache(codebook);

    SGVector<float64_t> query = binarize(outputs);
    SGVector<float64_t> L(codebook.num_cols);
    for (int32_t i=0; i < codebook.num_cols; ++i)
        L[i] = CECOCUtil::hamming_distance(query.vector, codebook.get_column_vector(i), query.vlen);

    SGVector<float64_t> res(codebook.num_cols);
    res.zero();
    // res = m_delta * L
    cblas_dgemv(CblasColMajor, CblasNoTrans, m_delta.num_cols, m_delta.num_cols,
            1, m_delta.matrix, m_delta.num_cols, L.vector, 1, 1, res.vector, 1);
    return SGVector<float64_t>::arg_max(res.vector, 1, res.vlen);
}

void CECOCIHDDecoder::update_delta_cache(const SGMatrix<int32_t> codebook)
{
    if (codebook.matrix == m_codebook.matrix)
        return; // memory address the same

    if (codebook.num_cols == m_codebook.num_cols && codebook.num_rows == m_codebook.num_rows)
    {
        bool the_same = true;
        for (int32_t i=0; i < codebook.num_rows && the_same; ++i)
            for (int32_t j=0; j < codebook.num_cols && the_same; ++j)
                if (codebook(i,j) != m_codebook(i,j))
                    the_same = false;
        if (the_same)
            return; // no need to update delta
    }

    m_codebook = codebook; // operator=
    m_delta = SGMatrix<float64_t>(codebook.num_cols, codebook.num_cols);
    m_delta.zero();
    for (int32_t i=0; i < codebook.num_cols; ++i)
    {
        for (int32_t j=i+1; j < codebook.num_cols; ++j)
        {
            m_delta(i, j) = m_delta(j, i) =
                CECOCUtil::hamming_distance(codebook.get_column_vector(i), codebook.get_column_vector(j), codebook.num_rows);
        }
    }

    // compute inverse of delta
    SGVector<int32_t> IPIV(m_delta.num_cols);
    clapack_dgetrf(CblasColMajor, m_delta.num_cols, m_delta.num_cols, m_delta.matrix, m_delta.num_cols, IPIV.vector);
    clapack_dgetri(CblasColMajor, m_delta.num_cols, m_delta.matrix, m_delta.num_cols, IPIV.vector);
}

#endif // HAVE_LAPACK
