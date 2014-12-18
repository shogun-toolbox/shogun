/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#ifndef _MBKMEANS_H__
#define _MBKMEANS_H__	

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/clustering/KMeans.h>


namespace shogun
{
class CKMeans;

/** Implementation class for the mini batch KMeans */
class CKMeansMiniBatch : public CKMeans
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


	private:

		/** train k-means
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);


		/** mini-batch KMeans training method
		 *
		 * @param k parameter k
		 * @param distance distance
 		 * @param batch_size parameter batch size
		 * @param minib_iter parameter number of iterations
		 * @param mus cluster centers matrix (k columns) 
		 */
		void minibatch_KMeans();


		/* choose b integers between 0 and num-1
		 * 
		 */
		static SGVector<int32_t> mbchoose_rand(int32_t b, int32_t num);

		void init();		

	private:

		///batch size for mini-batch KMeans
		int32_t m_batch_size;

		///number of iterations for mini-batch KMeans
		int32_t m_minib_iter;

};
}
#endif
