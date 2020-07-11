/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Soeren Sonnenburg, Chiyuan Zhang, Wu Lin,
 *          Sergey Lisitsyn, Roman Votyakov, Heiko Strathmann, Yuyu Zhang,
 *          Tonmoy Saikia, Soumyajit De, Sanuj Sharma
 */

#ifndef GAUSSIANKERNEL_H
#define GAUSSIANKERNEL_H

#include <shogun/lib/config.h>
#include <shogun/kernel/ShiftInvariantKernel.h>

namespace shogun
{

class Features;
class DotFeatures;
namespace params {
	class GaussianWidthAutoInit;
}
/** @brief The well known Gaussian kernel (swiss army knife for SVMs) computed
 * on DotFeatures.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 *
 */
class GaussianKernel: public ShiftInvariantKernel
{
	friend class params::GaussianWidthAutoInit;
public:
	/** default constructor */
	GaussianKernel();

	/** constructor
	 *
	 * @param width width
	 */
	GaussianKernel(float64_t width);

	/** constructor
	 *
	 * @param size cache size
	 * @param width width
	 */
	GaussianKernel(int32_t size, float64_t width);

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @param width width
	 * @param size cache size
	 */
	GaussianKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r, float64_t width, int32_t size=10);

	/** destructor */
	~GaussianKernel() override;

	/** @param kernel is casted to GaussianKernel, error if not possible
	 * is SG_REF'ed
	 * @return casted GaussianKernel object
	 */
	static std::shared_ptr<GaussianKernel> obtain_from_generic(const std::shared_ptr<Kernel>& kernel);

	/** initialize kernel
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if initializing was successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

	/** clean up kernel */
	void cleanup() override;

	/** return what type of kernel we are
	 *
	 * @return kernel type GAUSSIAN
	 */
	EKernelType get_kernel_type() override
	{
		return K_GAUSSIAN;
	}

	/** @return feature type of distance used */
	EFeatureType get_feature_type() override
	{
		return F_ANY;
	}

	/** @return feature class of distance used */
	EFeatureClass get_feature_class() override
	{
		return C_ANY;
	}

	/** return the kernel's name
	 *
	 * @return name Gaussian
	 */
	const char* get_name() const override { return "GaussianKernel"; }

	/** set the kernel's width
	 *
	 * @param w kernel width
	 */
	void set_width(float64_t w);

	/** return the kernel's width
	 *
	 * @return kernel width
	 */
	float64_t get_width() const
	{
		return GaussianKernel::from_log_width(std::get<float64_t>(m_log_width));
	}

	/**
	 * Converts width to log_width.
	 *
	 * @param log_width the kernel log width
	 * @return the kernel width
	 */
	static float64_t from_log_width(float64_t log_width) noexcept
	{
		return std::exp(log_width * 2.0) * 2.0;
	}

	/**
	 * Converts log_width to width.
	 *
	 * @param width the kernel width
	 * @return the kernel log width
	 */
	static float64_t to_log_width(float64_t width) noexcept
	{
		return std::log(width / 2.0) / 2.0;
	}

	/** return derivative with respect to specified parameter
	 *
	 * @param param the parameter
	 * @param index the index of the element if parameter is a vector
	 *
	 * @return gradient with respect to parameter
	 */
	SGMatrix<float64_t> get_parameter_gradient(Parameters::const_reference param, index_t index=-1) override;

	/** Can (optionally) be overridden to post-initialize some member
	 * variables which are not PARAMETER::ADD'ed. Make sure that at first
	 * the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST is called.
	 *
	 *  @exception ShogunException Will be thrown if an error occurres.
	 */
	void load_serializable_post() override;

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

	/** compute the distance between features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed the distance
	 *
	 * Note that in GaussianKernel,
	 * kernel(idx_a, idx_b)=exp(-distance(idx_a, idx_b))
	 * \f[
	 * 	distance({\bf x},{\bf y})= \frac{||{\bf x}-{\bf y}||^2}{\tau}
	 * \f]
	 */
	float64_t distance(int32_t idx_a, int32_t idx_b) const override;

protected:
	/** width */
	AutoValue<float64_t> m_log_width = AutoValueEmpty{};
};

}
#endif /* _GAUSSIANKERNEL_H__ */
