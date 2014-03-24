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
#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{
class CKMeansMiniBatchImpl
{
	public:
		/** mini-batch KMeans training method
		 *
		 * @param k parameter k
		 * @param distance distance
 		 * @param batch_size parameter batch size
		 * @param minib_iter parameter number of iterations
		 * @param mus cluster centers matrix (k columns) 
		 */
		static void minibatch_KMeans(int32_t k, CDistance* distance, int32_t batch_size, int32_t minib_iter, SGMatrix<float64_t> mus);

	private:
		/* choose b integers between 0 and num-1
		 * 
		 */
		static SGVector<int32_t> mbchoose_rand(int32_t b, int32_t num);
};
}
#endif
