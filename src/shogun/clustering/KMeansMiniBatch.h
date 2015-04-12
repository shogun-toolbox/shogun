/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2014 Parijat Mazumdar
 * Written (W) 2014 Saurabh Mahindre
 *
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

#ifndef _MBKMEANS_H__
#define _MBKMEANS_H__	

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/clustering/KMeansBase.h>


namespace shogun
{
class CKMeansBase;

/** Class for the mini batch KMeans */
class CKMeansMiniBatch : public CKMeansBase
{
	public:
		/** default constructor */
		CKMeansMiniBatch();

		/** constructor
		 *
		 * @param k parameter k
		 * @param d distance
		 * @param kmeanspp true for using KMeans++ (default false)
		 */
		CKMeansMiniBatch(int32_t k, CDistance* d, bool kmeanspp=false);

		/** constructor for supplying initial centers
		 * @param k_i parameter k
		 * @param d_i distance
		 * @param centers_i initial centers for KMeans aloverride private method c++gorithm
		*/
		CKMeansMiniBatch(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i);
		virtual ~CKMeansMiniBatch();

		/** @return object name */
		virtual const char* get_name() const { return "KMeansMiniBatch"; }	

		/** set batch size for mini-batch KMeans
		 *
		 *@param b batch size int32_t(greater than 0)
		 */
		void set_mbKMeans_batch_size(int32_t b);

		/** get batch size for mini-batch KMeans
		 *
		 *@return batch size
		 */
		int32_t get_mbKMeans_batch_size() const;

		/** set no. of iterations for mini-batch KMeans
		 *
		 *@param t no. of iterations int32_t(greater than 0)
		 */
		void set_mbKMeans_iter(int32_t t);

		/** get no. of iterations for mini-batch KMeans
		 *
		 *@return no. of iterations
		 */
		int32_t get_mbKMeans_iter() const;

		/** set batch size and no. of iteration for mini-batch KMeans
		 *
		 *@param b batch size
		 *@param t no. of iterations
		 */
		void set_mbKMeans_params(int32_t b, int32_t t);


	protected:

		/** train k-means
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	private:
		void init();		

	private:

		///batch size for mini-batch KMeans
		int32_t m_batch_size;

		///number of iterations for mini-batch KMeans
		int32_t m_minib_iter;

};
}
#endif
