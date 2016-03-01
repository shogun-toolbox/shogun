#ifndef _GAUSSIANCOMPACTKERNEL_H___
#define _GAUSSIANCOMPACTKERNEL_H___

#include <shogun/kernel/GaussianKernel.h>

namespace shogun
{
  class CDotFeatures;
  /** @brief The well known Gaussian kernel (swiss army knife for SVMs) computed
   * on CDotFeatures.
   * TODO
   */

  class CGaussianCompactKernel: public CGaussianKernel
  {
    public:
      /** default constructor */
      CGaussianCompactKernel();
      /** constructor
       *
       * @param size cache size
       * @param width width
       */
      CGaussianCompactKernel(int32_t size, float64_t width);
      /** constructor
       *
       * @param l features of left-hand side
       * @param r features of right-hand side
       * @param width width
       * @param size cache size
       */
      CGaussianCompactKernel(CDotFeatures* l, CDotFeatures* r, float64_t width, int32_t size=10);
      /* destructor */
      virtual ~CGaussianCompactKernel();

      /** return what type of kernel we are
       *
       * @return kernel type GAUSSIAN
       */
      virtual EKernelType get_kernel_type() { return K_GAUSSIANCOMPACT; }

      /** return the kernel's name
       *
       * @return name GaussianCompactKernel
       */
      virtual const char* get_name() const { return "GaussianCompactKernel"; }
    protected:
      /** compute kernel function for features a and b
       * idx_{a,b} denote the index of the feature vectors
       * in the corresponding feature object
       *
       * @param idx_a index a
       * @param idx_b index b
       * @return computed kernel function at indices a,b
       */
      virtual float64_t compute(int32_t idx_a, int32_t idx_b);
  };
}
#endif /* _GAUSSIANCOMPACTKERNEL_H__ */
