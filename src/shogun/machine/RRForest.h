/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2016 Saurabh Mahindre
 */

#ifndef _RRFOREST_H__
#define _RRFOREST_H__

#include <shogun/lib/config.h>
#include <shogun/machine/BaggingMachine.h>

namespace shogun
{

/** @brief Random Rotation - Random forest. Feature space is randomly rotated prior to training the base learners.
 *
 *  See: Random Rotation Ensembles Journal of Machine Learning Research, Vol. 17, No. 4. (2016), pp. 1-26 by Rico Blaser, Piotr Fryzlewicz
 
 */
class CRRForest : public CBaggingMachine
{
public:
	/** constructor */
	CRRForest();

	/** constructor
	 *
	 * @param num_rand_feats number of attributes chosen randomly during node split in candidate trees
	 * @param num_bags number of trees in forest
	 */
	CRRForest(int32_t num_rand_feats, int32_t num_bags=10);

	/** constructor
	 *
	 * @param features training features
	 * @param labels training labels
	 * @param num_bags number of trees in forest
	 * @param num_rand_feats number of attributes chosen randomly during node split in candidate trees
	 */
	CRRForest(CFeatures* features, CLabels* labels, int32_t num_bags=10, int32_t num_rand_feats=0);

	/** constructor
	 *
	 * @param features training features
	 * @param labels training labels
	 * @param weights weights of training feature vectors
	 * @param num_bags number of trees in forest
	 * @param num_rand_feats number of attributes chosen randomly during node split in candidate trees
	 */
	CRRForest(CFeatures* features, CLabels* labels, SGVector<float64_t> weights, int32_t num_bags=10, int32_t num_rand_feats=0);

	/** destructor */
	virtual ~CRRForest();

	/** get name
	 *
	 * @return RRForest
	 */
	virtual const char* get_name() const { return "RRForest"; }

	/** machine is set to modified CART(RandomCART) and cannot be changed
	 *
	 * @param machine the machine to use for bagging
	 */
	virtual void set_machine(CMachine* machine);

	/** set weights
	 *
	 * @param weights of training feature vectors
	 */
	void set_weights(SGVector<float64_t> weights);

	/** get weights
	 *
	 * @return weights of training feature vectors
	 */
	SGVector<float64_t> get_weights() const;

	/** get problem type - multiclass classification or regression
	 *
	 * @return PT_MULTICLASS or PT_REGRESSION
	 */
	virtual EProblemType get_machine_problem_type() const;

	/** set problem type - multiclass classification or regression
	 *
	 * @param mode EProblemType PT_MULTICLASS or PT_REGRESSION
	 */
	void set_machine_problem_type(EProblemType mode);

	/** set number of random features to be chosen during node splits
	 *
	 * @param rand_featsize number of randomly chosen features during each node split
	 */
	void set_num_random_features(int32_t rand_featsize);

	/** get number of random features to be chosen during node splits
	 *
	 * @return number of randomly chosen features during each node split
	 */
	int32_t get_num_random_features() const;

protected:

	virtual bool train_machine(CFeatures* data=NULL);
	/** sets parameters of CARTree - sets machine labels and weights here
	 *
	 * @param m machine
	 * @param idx indices of training vectors chosen in current bag
	 */
	virtual void set_machine_parameters(CMachine* m, SGVector<index_t> idx);

private:
	/** initialize parameters */
	void init();

private:
	/** weights */
	SGVector<float64_t> m_weights;

};
} /* namespace shogun */
#endif /* _RRFOREST_H__ */

