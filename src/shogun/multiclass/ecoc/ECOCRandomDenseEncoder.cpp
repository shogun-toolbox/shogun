/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Chiyuan Zhang
 */

#include <algorithm>
#include <limits>

#include <shogun/multiclass/ecoc/ECOCRandomDenseEncoder.h>
#include <shogun/multiclass/ecoc/ECOCUtil.h>
#include <shogun/mathematics/UniformRealDistribution.h>

using namespace shogun;

ECOCRandomDenseEncoder::ECOCRandomDenseEncoder() : RandomMixin<ECOCEncoder>()
{
	init();
}

ECOCRandomDenseEncoder::ECOCRandomDenseEncoder(int32_t maxiter, int32_t codelen, float64_t pposone)
: RandomMixin<ECOCEncoder>()
{
    if (!check_probability(pposone))
        error("invalid probability of +1");

    init();
	m_maxiter = maxiter;
	m_codelen = codelen;
	m_pposone = pposone;
}

void ECOCRandomDenseEncoder::init()
{
	m_maxiter = 10000;
	m_codelen = 0;
	m_pposone = 0.5;
    SG_ADD(&m_maxiter, "maxiter", "max number of iterations");
    SG_ADD(&m_codelen, "codelen", "code length");
    SG_ADD(&m_pposone, "pposone", "probability of +1");
}

void ECOCRandomDenseEncoder::set_probability(float64_t pposone)
{
    if (!check_probability(pposone))
        error("probability of 0, +1 and -1 must sum to one");

    m_pposone = pposone;
}

SGMatrix<int32_t> ECOCRandomDenseEncoder::create_codebook(int32_t num_classes)
{
    int32_t codelen = m_codelen;
    if (codelen <= 0)
        codelen = get_default_code_length(num_classes);


    SGMatrix<int32_t> best_codebook(codelen, num_classes, true);
    int32_t best_dist = 0;

    SGMatrix<int32_t> codebook(codelen, num_classes);
    UniformRealDistribution<float64_t> uniform_real_dist(0.0, 1.0);
    int32_t n_iter = 0;
    while (true)
    {
        // fill codebook
        codebook.zero();
        for (int32_t i=0; i < codelen; ++i)
        {
            for (int32_t j=0; j < num_classes; ++j)
            {
                float64_t randval = uniform_real_dist(m_prng);
                if (randval > m_pposone)
                    codebook(i, j) = -1;
                else
                    codebook(i, j) = +1;
            }
        }

        bool valid = true;
        for (int32_t i=0; i < codelen; ++i)
        {
            bool p1_occur = false, n1_occur = false;
            for (int32_t j=0; j < num_classes; ++j)
                if (codebook(i, j) == 1)
                    p1_occur = true;
                else if (codebook(i, j) == -1)
                    n1_occur = true;

            if (!p1_occur || !n1_occur)
            {
                valid = false;
                break;
            }
        }

        if (valid)
        {
            // see if this is a better codebook
            // compute the minimum pairwise code distance
            int32_t min_dist = std::numeric_limits<int32_t>::max();
            for (int32_t i=0; i < num_classes; ++i)
            {
                for (int32_t j=i+1; j < num_classes; ++j)
                {
                    int32_t dist = ECOCUtil::hamming_distance(codebook.get_column_vector(i),
                            codebook.get_column_vector(j), codelen);
                    if (dist < min_dist)
                        min_dist = dist;
                }
            }

            if (min_dist > best_dist)
            {
                best_dist = min_dist;
                std::copy(codebook.matrix, codebook.matrix + codelen*num_classes,
                        best_codebook.matrix);
            }
        }

        if (++n_iter >= m_maxiter)
            if (best_dist > 0) // already obtained a good codebook
                break;
    }

    return best_codebook;
}
