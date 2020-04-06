#ifndef _GAUSSIANCOMPACTKERNEL_H___
#define _GAUSSIANCOMPACTKERNEL_H___

#include <shogun/kernel/GaussianKernel.h>

namespace shogun
{
class DotFeatures;
/** @brief The compact version as given in Bart Hamers' thesis
 * <i>Kernel Models for Large Scale Applications</i>
 * (Eq. 4.10) is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= max(0, (1-\frac{||{\bf x}-{\bf x'}||}{3\tau})^v)) *
 * exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 *
 */

class GaussianCompactKernel: public GaussianKernel
{
    public:
        /** default constructor */
        GaussianCompactKernel();

        /** constructor
         *
         * @param size cache size
         * @param width width
         */
        GaussianCompactKernel(int32_t size, float64_t width);

        /** constructor
         *
         * @param l features of left-hand side
         * @param r features of right-hand side
         * @param width width
         * @param size cache size
         */
        GaussianCompactKernel(std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r,
                               float64_t width, int32_t size=10);

        /* destructor */
        ~GaussianCompactKernel() override;

        /** return what type of kernel we are
         *
         * @return kernel type GAUSSIAN
         */
        EKernelType get_kernel_type() override
        {
            return K_GAUSSIANCOMPACT;
        }

        /** return the kernel's name
         *
         * @return name GaussianCompactKernel
         */
        const char* get_name() const override
        {
            return "GaussianCompactKernel";
        }

    protected:
        /** compute kernel function for features a and b
         * idx_{a,b} denote the index of the feature vectors
         * in the corresponding feature object
         *
         * @param idx_a index a
         * @param idx_b index b
         * @return computed kernel function at indices a,b
         */
        float64_t compute(int32_t idx_a, int32_t idx_b) override;

};
}
#endif /* _GAUSSIANCOMPACTKERNEL_H__ */
