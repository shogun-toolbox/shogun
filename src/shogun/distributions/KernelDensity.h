/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef _KERNELDENSITY_H__
#define _KERNELDENSITY_H__

#include <shogun/lib/config.h>
#include <shogun/distributions/Distribution.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/multiclass/tree/NbodyTree.h>

namespace shogun
{
enum EEvaluationMode
{
	EM_KDTREE_SINGLE,
	EM_BALLTREE_SINGLE
};

/** @brief This class implements the kernel density estimation technique. Kernel density estimation is a non-parametric
 * way to estimate an unknown pdf. The pdf at a query point given finite training samples is calculated using the
 * following formula : \\
 * \f$pdf(x')= \frac{1}{nh} \sum_{i=1}^n K(\frac{||x-x_i||}{h})\f$ \\
 * K() in the above formula is called the kernel function and is controlled by the parameter h called kernel bandwidth.
 * Presently, this class supports only Gaussian kernel which can be used with either Euclidean distance or Manhattan
 * distance. This class makes use of 2 tree structures KD-tree and Ball tree for fast calculation. KD-trees are
 * faster than ball trees at lower dimensions. In case of high dimensional data, ball tree tends to out-perform KD-tree.
 * By default, the class used is Ball tree.
 */
class CKernelDensity : public CDistribution
{
public :
	/** Constructor
	 *
	 * @param bandwidth bandwidth of the kernel
	 * @param kernel_type type of kernel used
	 * @param dist distance metric used
	 * @param eval evaluation mode
	 * @param leaf_size min allowed vectors in leaves of the underlying tree
	 * @param atol absolute tolerance
	 * @param rtol relative tolerance   
	 */
	CKernelDensity(float64_t bandwidth=1.0, EKernelType kernel_type=K_GAUSSIAN, EDistanceType dist=D_EUCLIDEAN, EEvaluationMode eval=EM_BALLTREE_SINGLE, int32_t leaf_size=1, float64_t atol=0, float64_t rtol=0);

	/** destructor */
	~CKernelDensity();

	/** return class name
	 *
	 * @return KernelDensity
	 */
	virtual const char* get_name() const { return "KernelDensity"; } 

	/** form tree using input points
	 * 
	 * @param data data points to be used for density estimation
	 * @return true 
	 */
	virtual bool train(CFeatures* data=NULL);

	/** compute kde for given test points
	 * 
	 * @param test data points at which kernel density is to be evaluated
	 * @return log of estimated kernel density velues at given test points
	 */
	SGVector<float64_t> get_log_density(CDenseFeatures<float64_t>* test);

	/** return number of model parameters
	 * NOT IMPLEMENTED
	 *
	 * @return number of model parameters
	 */
	virtual int32_t get_num_model_parameters();

	/** return log model parameter
	 * NOT IMPLEMENTED
	 *
	 * @param num_param index number of the parameter
	 * @return log of model parameter
	 */
	virtual float64_t get_log_model_parameter(int32_t num_param);

	/** return log derivative
	 * NOT IMPLEMENTED
	 * 
	 * @param num_param index number of the parameter
	 * @param num_example index number of example
	 * @return log of derivative of example
	 */
	virtual float64_t get_log_derivative(int32_t num_param, int32_t num_example);

	/** return log likelihood of example
	 * NOT IMPLEMENTED
	 *
	 * @param num_example index number of example
	 * @return log likelihood of example
	 */
	virtual float64_t get_log_likelihood_example(int32_t num_example);

	/** returns norm of a given kernel
	 *
	 * @param kernel kernel whose norm is to be calculated
	 * @param width kernel bandwidth
	 * @param dim kernel dimension
	 * @return log of norm of kernel
	 */
	inline static float64_t log_norm(EKernelType kernel, float64_t width, int32_t dim)
	{
		switch(kernel)
		{
			case K_GAUSSIAN:
			{
				return -0.5*dim* CMath::log(2*CMath::PI)-dim*CMath::log(width);
				break;
			}
			default:
				SG_SPRINT("kernel type not recognized\n");
		}

		return 0.0;
	}

	/** returns kernel value
	 *
	 * @param kernel kernel type
	 * @param dist distance
	 * @param width kernel width
	 * @return log of kernel
	 */
	inline static float64_t log_kernel(EKernelType kernel, float64_t dist, float64_t width)
	{
		switch(kernel)
		{
			case K_GAUSSIAN:
			{
				return -0.5*dist*dist/(width*width);
				break;
			}
			default:
				SG_SPRINT("kernel type not recognized\n");
		}

		return 0.0;
	}

private:
	/** initialize */
	void init();

private :
	/** bandwidth */
	float64_t m_bandwidth;

	/** leaf size */
	int32_t m_leaf_size;

	/** absolute tolerance */
	float64_t m_atol;

	/** relative tolerance */
	float64_t m_rtol;

	/** evaluation mode */
	EEvaluationMode m_eval;

	/** kernel */
	EKernelType m_kernel_type;

	/** distance metric */
	EDistanceType m_dist;

	/** Tree */
	CNbodyTree* tree;
};
} /* shogun */

#endif /* _KERNELDENSITY_H__ */ 