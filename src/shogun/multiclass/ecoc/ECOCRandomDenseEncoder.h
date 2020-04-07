/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang, Sergey Lisitsyn, Viktor Gal
 */

#ifndef ECOCRANDOMDENSEENCODER_H__
#define ECOCRANDOMDENSEENCODER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCEncoder.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** Generate random ECOC codebook containing +1 and -1, and
 * select the best one.
 */
class ECOCRandomDenseEncoder: public RandomMixin<ECOCEncoder>
{
public:
	/** Default constructor */
	ECOCRandomDenseEncoder();

    /** constructor
     * @param maxiter max number of iterations
     * @param codelen code length, if set to zero, will be computed automatically via get_default_code_length
     * @param pposone probability of +1
     *
     * @see get_default_code_length
     */
    ECOCRandomDenseEncoder(int32_t maxiter, int32_t codelen, float64_t pposone);

    /** destructor */
    ~ECOCRandomDenseEncoder() override {}

    /** set probability
     * @param pposone probability of +1
     */
    void set_probability(float64_t pposone);

    /** get name */
    const char* get_name() const override { return "ECOCRandomDenseEncoder"; }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    SGMatrix<int32_t> create_codebook(int32_t num_classes) override;

    /** get default code length
     * @param num_classes number of classes
     *
     * In Dense Random Coding, 10 * log(num_classes) is suggested as code length.
     * See
     *
     *   E. Allwein, R. Schapire, and Y. Singer. Reducing multiclass to binary: A unifying approach
     *   for margin classifiers. Journal of Machine Learning Research, 1:113-141, 2002.
     */
    int32_t get_default_code_length(int32_t num_classes) const
    {
		return static_cast<int32_t>(
			Math::round(10 * std::log(static_cast<float64_t>(num_classes))));
	}

protected:
    int32_t   m_maxiter; ///< max number of iterations
    int32_t   m_codelen; ///< code length
    float64_t m_pposone; ///< probability of +1

private:
    /** ensure probability sum to one
     * @param pposone probability of +1
     */
    bool check_probability(float64_t pposone)
    {
        if (pposone >= 0.999 || pposone <= 0.0001)
            return false;
        return true;
    }

    /** init parameters */
    void init();
};

} // namespace shogun

#endif /* end of include guard: ECOCRANDOMDENSEENCODER_H__ */

