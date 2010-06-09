/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WEIGHTEDDEGREESTRINGKERNEL_H___
#define _WEIGHTEDDEGREESTRINGKERNEL_H___

#include "lib/common.h"
#include "lib/Trie.h"
#include "kernel/StringKernel.h"
#include "kernel/MultitaskKernelMklNormalizer.h"
#include "features/StringFeatures.h"






namespace shogun
{

enum EWDKernType
{
	E_WD=0,
	E_EXTERNAL=1,

	E_BLOCK_CONST=2,
	E_BLOCK_LINEAR=3,
	E_BLOCK_SQPOLY=4,
	E_BLOCK_CUBICPOLY=5,
	E_BLOCK_EXP=6,
	E_BLOCK_LOG=7,
	E_BLOCK_EXTERNAL=8
};


/** @brief The Weighted Degree String kernel.
 *
 *  The WD kernel of order d compares two sequences \f${\bf x}\f$ and
 *  \f${\bf x'}\f$ of length L by summing all contributions of k-mer matches of
 *  lengths \f$k\in\{1,\dots,d\}\f$, weighted by coefficients \f$\beta_k\f$. It
 *  is defined as
 *  \f[
 *      k({\bf x},{\bf x'})=\sum_{k=1}^d\beta_k\sum_{l=1}^{L-k+1}I({\bf u}_{k,l}({\bf x})={\bf u}_{k,l}({\bf x'})).
 *  \f]
 *      Here, \f${\bf u}_{k,l}({\bf x})\f$ is the string of length k starting at position
 *      l of the sequence \f${\bf x}\f$ and \f$I(\cdot)\f$ is the indicator function
 *      which evaluates to 1 when its argument is true and to 0
 *      otherwise.
 */
class CWeightedDegreeStringKernel: public CStringKernel<char>
{
	public:

		/** constructor
		 *
		 * @param degree degree
		 * @param type weighted degree kernel type
		 */
		CWeightedDegreeStringKernel(int32_t degree, EWDKernType type=E_WD);

		/** constructor
		 *
		 * @param weights kernel's weights
		 * @param degree degree
		 */
		CWeightedDegreeStringKernel(float64_t* weights, int32_t degree);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		CWeightedDegreeStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t degree);

		virtual ~CWeightedDegreeStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** get WD kernel weighting type
		 *
		 * @return weighting type
		 *
		 *
		 * \sa EWDKernType
		 */
		EWDKernType get_type() const
		{
			return type;
		}

		/** get degree of WD kernel
		 *
		 * @return degree of the kernel
		 */
		int32_t get_degree() const
		{
			return degree;
		}

		/** get the number of mismatches that are allowed in WD kernel
		 * computation
		 *
		 * @return number of mismatches
		 */
		int32_t get_max_mismatch() const
		{
			return max_mismatch;
		}

		/** return what type of kernel we are
		 *
		 * @return kernel type WEIGHTEDDEGREE
		 */
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREE; }

		/** return the kernel's name
		 *
		 * @return name WeightedDegree
		 */
		virtual const char* get_name() const { return "WeightedDegree"; }

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param alphas alphas
		 * @return if initializing was successful
		 */
		inline virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t* alphas)
		{
			return init_optimization(count, IDX, alphas, -1);
		}

		/** initialize optimization
		 * do initialization for tree_num up to upto_tree, use
		 * tree_num=-1 to construct all trees
		 *
		 * @param count count
		 * @param IDX IDX
		 * @param alphas alphas
		 * @param tree_num which tree
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t* alphas, int32_t tree_num);

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		virtual bool delete_optimization();

		/** compute optimized
	 	*
	 	* @param idx index to compute
	 	* @return optimized value at given index
	 	*/
		virtual float64_t compute_optimized(int32_t idx)
		{
			if (get_is_initialized())
				return compute_by_tree(idx);

			SG_ERROR( "CWeightedDegreeStringKernel optimization not initialized\n");
			return 0;
		}

		/** helper for compute batch
		 *
		 * @param p thread parameter
		 */
		static void* compute_batch_helper(void* p);

		/** compute batch
		 *
		 * @param num_vec number of vectors
		 * @param vec_idx vector index
		 * @param target target
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @param factor factor
		 */
		virtual void compute_batch(
			int32_t num_vec, int32_t* vec_idx, float64_t* target,
			int32_t num_suppvec, int32_t* IDX, float64_t* alphas,
			float64_t factor=1.0);

		/** clear normal
		 * subkernel functionality
		 */
		inline virtual void clear_normal()
		{
			if (get_is_initialized())
			{

				if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
					SG_ERROR("not implemented");

				tries->delete_trees(max_mismatch==0);
				set_is_initialized(false);
			}
		}

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		inline virtual void add_to_normal(int32_t idx, float64_t weight)
		{

			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				SG_ERROR("not implemented");

			if (max_mismatch==0)
				add_example_to_tree(idx, weight);
			else
				add_example_to_tree_mismatch(idx, weight);

			set_is_initialized(true);
		}

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		inline virtual int32_t get_num_subkernels()
		{
			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				return ((CMultitaskKernelMklNormalizer*)normalizer)->get_num_betas();
			if (position_weights!=NULL)
				return (int32_t) ceil(1.0*seq_length/mkl_stepsize) ;
			if (length==0)
				return (int32_t) ceil(1.0*get_degree()/mkl_stepsize);
			return (int32_t) ceil(1.0*get_degree()*length/mkl_stepsize) ;
		}

		/** compute by subkernel
		 *
		 * @param idx index
		 * @param subkernel_contrib subkernel contribution
		 */
		inline void compute_by_subkernel(
			int32_t idx, float64_t * subkernel_contrib)
		{

			if (get_is_initialized())
			{

				if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
					SG_ERROR("not implemented");

				compute_by_tree(idx, subkernel_contrib);
				return ;
			}

			SG_ERROR( "CWeightedDegreeStringKernel optimization not initialized\n");
		}

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
		inline const float64_t* get_subkernel_weights(int32_t& num_weights)
		{

			num_weights = get_num_subkernels();

			delete[] weights_buffer ;
			weights_buffer = new float64_t[num_weights];

			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				for (int32_t i=0; i<num_weights; i++)
					weights_buffer[i] = ((CMultitaskKernelMklNormalizer*)normalizer)->get_beta(i);
			else if (position_weights!=NULL)
				for (int32_t i=0; i<num_weights; i++)
					weights_buffer[i] = position_weights[i*mkl_stepsize];
			else
				for (int32_t i=0; i<num_weights; i++)
					weights_buffer[i] = weights[i*mkl_stepsize];

			return weights_buffer;
		}

		/** set subkernel weights
		 *
		 * @param weights2 weights
		 * @param num_weights2 number of weights
		 */
		inline void set_subkernel_weights(
			float64_t* weights2, int32_t num_weights2)
		{
			int32_t num_weights = get_num_subkernels();
			if (num_weights!=num_weights2)
				SG_ERROR( "number of weights do not match\n");


			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				for (int32_t i=0; i<num_weights; i++)
					((CMultitaskKernelMklNormalizer*)normalizer)->set_beta(i, weights2[i]);
			else if (position_weights!=NULL)
			{
				for (int32_t i=0; i<num_weights; i++)
				{
					for (int32_t j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<seq_length)
							position_weights[i*mkl_stepsize+j] = weights2[i];
					}
				}
			}
			else if (length==0)
			{
				for (int32_t i=0; i<num_weights; i++)
				{
					for (int32_t j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<get_degree())
							weights[i*mkl_stepsize+j] = weights2[i];
					}
				}
			}
			else
			{
				for (int32_t i=0; i<num_weights; i++)
				{
					for (int32_t j=0; j<mkl_stepsize; j++)
					{
						if (i*mkl_stepsize+j<get_degree()*length)
							weights[i*mkl_stepsize+j] = weights2[i];
					}
				}
			}
		}

		/** set the current kernel normalizer
		 *
		 * @return if successful
		 */
		virtual bool set_normalizer(CKernelNormalizer* normalizer_) {

			if (normalizer_ && strcmp(normalizer_->get_name(),"MultitaskKernelTreeNormalizer")==0) {
				unset_property(KP_LINADD);
				unset_property(KP_BATCHEVALUATION);
			}
			else
			{
				set_property(KP_LINADD);
				set_property(KP_BATCHEVALUATION);
			}


			return CStringKernel<char>::set_normalizer(normalizer_);

		}

		// other kernel tree operations
		/** compute abs weights
		 *
		 * @param len len
		 * @return computed abs weights
		 */
		float64_t *compute_abs_weights(int32_t & len);

		/** compute by tree
		 *
		 * @param idx index
		 * @param LevelContrib level contribution
		 * @return computed value
		 */
		void compute_by_tree(int32_t idx, float64_t *LevelContrib);

		/** check if tree is initialized
		 *
		 * @return if tree is initialized
		 */
		bool is_tree_initialized() { return tree_initialized; }

		/** get degree weights
		 *
		 * @param d degree weights will be stored here
		 * @param len number of degree weights will be stored here
		 */
		inline float64_t *get_degree_weights(int32_t& d, int32_t& len)
		{
			d=degree;
			len=length;
			return weights;
		}

		/** get weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return weights
		 */
		inline float64_t *get_weights(int32_t& num_weights)
		{

			if (normalizer && normalizer->get_normalizer_type()==N_MULTITASK)
				SG_ERROR("not implemented");

			if (position_weights!=NULL)
			{
				num_weights = seq_length ;
				return position_weights ;
			}
			if (length==0)
				num_weights = degree ;
			else
				num_weights = degree*length ;
			return weights;
		}

		/** get position weights
		 *
		 * @param len number of position weights will be stored here
		 * @return position weights
		 */
		inline float64_t *get_position_weights(int32_t& len)
		{
			len=seq_length;
			return position_weights;
		}

		/** set wd weights
		 *
		 * @param type weighted degree kernel type
		 * @return if setting was successful
		 */
		bool set_wd_weights_by_type(EWDKernType type);

		/** set wd weights
		 *
		 * @param p_weights new eights
		 * @param d degree
		 * @return if setting was successful
		 */
		void set_wd_weights(float64_t* p_weights, int32_t d)
		{
			set_weights(p_weights,d,0);
		}

		/** set weights
		 *
		 * @param weights new weights
		 * @param d degree
		 * @param len number of weights
		 */
		bool set_weights(float64_t* weights, int32_t d, int32_t len);

		/** set position weights
		 *
		 * @param pws new position weights
		 * @param len number of position weights
		 * @return if setting was successful
		 */
		bool set_position_weights(float64_t* pws, int32_t len=0);

		/** initialize block weights
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights();

		/** initialize block weights from weighted degree
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_from_wd();

		/** initialize block weights from external weighted degree
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_from_wd_external();

		/** initialize block weights constant
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_const();

		/** initialize block weights linear
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_linear();

		/** initialize block weights squared polynomial
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_sqpoly();

		/** initialize block weights cubic polynomial
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_cubicpoly();

		/** initialize block weights exponential
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_exp();

		/** initialize block weights logarithmic
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_log();

		/** initialize block weights external
		 *
		 * @return if initialization was successful
		 */
		bool init_block_weights_external();

		/** delete position weights
		 *
		 * @return if deleting was successful
		 */
		bool delete_position_weights()
		{
			delete[] position_weights;
			position_weights=NULL;
			return true;
		}

		/** set maximum mismatch
		 *
		 * @param max new maximum mismatch
		 * @return if setting was successful
		 */
		bool set_max_mismatch(int32_t max);

		/** get maximum mismatch
		 *
		 * @return maximum mismatch
		 */
		inline int32_t get_max_mismatch() { return max_mismatch; }

		/** set degree
		 *
		 * @param deg new degree
		 * @return if setting was successful
		 */
		inline bool set_degree(int32_t deg) { degree=deg; return true; }

		/** get degree
		 *
		 * @return degree
		 */
		inline int32_t get_degree() { return degree; }

		/** set if block computation shall be performed
		 *
		 * @param block if block computation shall be performed
		 * @return if setting was successful
		 */
		inline bool set_use_block_computation(bool block)
		{
			block_computation=block;
			return true;
		}

		/** check if block computation is performed
		 *
		 * @return if block computation is performed
		 */
		inline bool get_use_block_computation() { return block_computation; }

		/** set MKL steps ize
		 *
		 * @param step new step size
		 * @return if setting was successful
		 */
		inline bool set_mkl_stepsize(int32_t step)
		{
			if (step<1)
				SG_ERROR("Stepsize must be a positive integer\n");
			mkl_stepsize=step;
			return true;
		}

		/** get MKL step size
		 *
		 * @return MKL step size
		 */
		inline int32_t get_mkl_stepsize() { return mkl_stepsize; }

		/** set which degree
		 *
		 * @param which which degree
		 * @return if setting was successful
		 */
		inline bool set_which_degree(int32_t which)
		{
			which_degree=which;
			return true;
		}

		/** get which degree
		 *
		 * @return which degree
		 */
		inline int32_t get_which_degree() { return which_degree; }

	protected:
		/** create emtpy tries */
		void create_empty_tries();

		/** add example to tree
		 *
		 * @param idx index
		 * @param weight weight
		 */
		void add_example_to_tree(int32_t idx, float64_t weight);

		/** add example to single tree
		 *
		 * @param idx index
		 * @param weight weight
		 * @param tree_num which tree
		 */
		void add_example_to_single_tree(
			int32_t idx, float64_t weight, int32_t tree_num);

		/** add example to tree mismatch
		 *
		 * @param idx index
		 * @param weight weight
		 */
		void add_example_to_tree_mismatch(int32_t idx, float64_t weight);

		/** add example to single tree mismatch
		 *
		 * @param idx index
		 * @param weight weight
		 * @param tree_num which tree
		 */
		void add_example_to_single_tree_mismatch(
			int32_t idx, float64_t weight, int32_t tree_num);

		/** compute by tree
		 *
		 * @param idx index
		 * @return computed value
		 */
		float64_t compute_by_tree(int32_t idx);

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b);

		/** compute with mismatch
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		float64_t compute_with_mismatch(
			char* avec, int32_t alen, char* bvec, int32_t blen);

		/** compute without mismatch
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		float64_t compute_without_mismatch(
			char* avec, int32_t alen, char* bvec, int32_t blen);

		/** compute without mismatch matrix
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		float64_t compute_without_mismatch_matrix(
			char* avec, int32_t alen, char* bvec, int32_t blen);

		/** compute using block
		 *
		 * @param avec vector a
		 * @param alen length of vector a
		 * @param bvec vector b
		 * @param blen length of vector b
		 * @return computed value
		 */
		float64_t compute_using_block(char* avec, int32_t alen,
			char* bvec, int32_t blen);

		/** remove lhs from kernel */
		virtual void remove_lhs();


#ifdef HAVE_BOOST_SERIALIZATION




    private:

        // serialization needs to split up in save/load because
        // the serialization of pointers to natives (int* & friends)
        // requires a workaround
        friend class ::boost::serialization::access;

        //  friend std::ostream & operator<<(std::ostream &os, const CWeightedDegreeStringKernel &gp);
        //template<class Archive>
        //friend void ::boost::serialization::save_construct_data(Archive & ar, const CWeightedDegreeStringKernel* t, const unsigned int file_version);
        template<class Archive>
            void save(Archive & ar, const unsigned int archive_version) const
            {

                SG_DEBUG("archiving CWeightedDegreeStringKernel\n");

                ar & ::boost::serialization::base_object<CStringKernel<char> >(*this);


                ///degree*length weights
                ///length must match seq_length if != 0
                ar & mkl_stepsize ;
                //ar & degree;
                ar & length;
                ar & max_mismatch ;

                //for (int32_t i=0; i<degree*(1+max_mismatch); i++)
                //    ar & weights[i];

                //TODO how long?
                //float64_t* position_weights ;
                //float64_t* weights_buffer ;

                ar & seq_length ;

                ar & initialized ;
                ar & block_computation;
                //ar & use_normalization ;

                //ar & normalization_const;

                ar & num_block_weights_external;
                for (int32_t i=0; i < num_block_weights_external; ++i)
                {
                    ar & block_weights_external[i];
                }

                //TODO how long
                //float64_t* block_weights;
                //ar & type;
                ar & which_degree;

                //TODO implement
                //CTrie<DNATrie> tries ;
                //ar & tree_initialized ;


                //CWeightedDegreeStringKernel* tmp = const_cast<CWeightedDegreeStringKernel*>(this);
                //tmp->create_empty_tries();
                //create_empty_tries();

                SG_DEBUG("done with CWeightedDegreeStringKernel\n");

            }

        template<class Archive>
            void load(Archive & ar, const unsigned int archive_version)
            {
                SG_DEBUG("archiving CWeightedDegreeStringKernel\n");

                ar & ::boost::serialization::base_object<CStringKernel<char> >(*this);


                ///degree*length weights
                ///length must match seq_length if != 0
                ar & mkl_stepsize ;
                ar & degree;
                ar & length;
                ar & max_mismatch ;

                //weights=new float64_t[degree*(1+max_mismatch)];
                //for (int32_t i=0; i<degree*(1+max_mismatch); i++)
                //    ar & weights[i];


                //TODO how long?
                //float64_t* position_weights ;
                //float64_t* weights_buffer ;

                ar & seq_length ;

                ar & initialized ;
                ar & block_computation;
                //ar & use_normalization ;

                //ar & normalization_const;

                ar & num_block_weights_external;
                //float64_t* block_weights_external;
                block_weights_external = new float64_t[num_block_weights_external];
                for (int32_t i=0; i < num_block_weights_external; ++i)
                {
                    ar & block_weights_external[i];
                }

                //TODO how long
                //float64_t* block_weights;
                //ar & type;
                ar & which_degree;

                //TODO implement
                //CTrie<DNATrie> tries ;
                //ar & tree_initialized ;

                SG_DEBUG("done with CWeightedDegreeStringKernel\n");

            }

        GLOBAL_BOOST_SERIALIZATION_SPLIT_MEMBER();


    public:

        virtual std::string toString() const
        {
            std::ostringstream s;

            ::boost::archive::text_oarchive oa(s);

            oa << *this;

            return s.str();
        }

        virtual void fromString(std::string str)
        {

            std::istringstream is(str);

            ::boost::archive::text_iarchive ia(is);

            ia >> *this;

        }

#endif //HAVE_BOOST_SERIALIZATION


	protected:
		/** degree*length weights
		 *length must match seq_length if != 0
		 */
		float64_t* weights;
		/** position weights */
		float64_t* position_weights;
		/** weights buffer */
		float64_t* weights_buffer;
		/** MKL step size */
		int32_t mkl_stepsize;
		/** degree */
		int32_t degree;
		/** length */
		int32_t length;

		/** maximum mismatch */
		int32_t max_mismatch;
		/** sequence length */
		int32_t seq_length;

		/** if kernel is initialized */
		bool initialized;

		/** if block computation is used */
		bool block_computation;

		/** number of external block weights */
		int32_t num_block_weights_external;
		/** external block weights */
		float64_t* block_weights_external;

		/** (internal) block weights */
		float64_t* block_weights;
		/** WeightedDegree kernel type */
		EWDKernType type;
		/** which degree */
		int32_t which_degree;

		/** tries */
		CTrie<DNATrie>* tries;

		/** if tree is initialized */
		bool tree_initialized;

		/** alphabet of features */
		CAlphabet* alphabet;
};

}



#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/export.hpp>

#endif //HAVE_BOOST_SERIALIZATION


#ifdef HAVE_BOOST_SERIALIZATION

namespace boost
{
	namespace serialization
	{
		template<class Archive>
			//inline void save_construct_data(Archive & ar, const shogun::CWeightedDegreeStringKernel* const t, const unsigned int file_version)
			inline void save_construct_data(Archive & ar, shogun::CWeightedDegreeStringKernel* t, const unsigned int file_version)
			{

				std::cout << "saving WDK from non-defaultconstruct data works" << std::endl;

				//CWeightedDegreeStringKernel(INT size, EWDKernType type, INT degree, INT max_mismatch, bool use_normalization=true, bool block_computation=false, INT mkl_stepsize=1, INT which_deg=-1);

				//ar << t->cache_size;

				ar << t->type;

				ar << t->degree;

				//ar << t->max_mismatch;
				/*

				ar.register_type(static_cast<shogun::CStringFeatures<char> *>(NULL));


				const shogun::CStringFeatures<char>* const lhs = dynamic_cast<shogun::CStringFeatures<char>* >(const_cast<shogun::CWeightedDegreeStringKernel*>(t)->get_lhs());

				const shogun::CStringFeatures<char>* const rhs = dynamic_cast<shogun::CStringFeatures<char>* >(const_cast<shogun::CWeightedDegreeStringKernel*>(t)->get_rhs());

				//CStringFeatures<char>* lhs = (CStringFeatures<char>*) (const_cast<CWeightedDegreeStringKernel*>(t)->get_lhs());
				//CStringFeatures<char>* rhs = (CStringFeatures<char>*) (const_cast<CWeightedDegreeStringKernel*>(t)->get_rhs());

				//    const CFeatures* const lhs = t->get_lhs();
				//    const CFeatures* const rhs = t->get_rhs();

				ar << lhs;
				ar << rhs;

				//ar << dynamic_cast<CStringFeatures<char>*>(rhs);
				//ar << t->get_lhs();
				//ar << t->get_rhs();
				*/
				std::cout << "done saving WDK from non-defaultconstruct data" << std::endl;

			}

		template<class Archive>
			inline void load_construct_data(Archive & ar, shogun::CWeightedDegreeStringKernel * t, const unsigned int file_version)
			{

				std::cout << "loading WDK from non-defaultconstruct data" << std::endl;



				shogun::EWDKernType type;
				int32_t degree;

				ar >> type;
				ar >> degree;
				/*
				int32_t size;
				int32_t max_mismatch;

				ar >> size;
				ar >> type;
				ar >> degree;
				ar >> max_mismatch;

				//      ::new(t)CWeightedDegreeStringKernel(size, type, degree, max_mismatch);

				shogun::CStringFeatures<char>* lhs;
				shogun::CStringFeatures<char>* rhs;


				ar >> lhs;
				ar >> rhs;

				::new(t)shogun::CWeightedDegreeStringKernel(lhs, rhs, degree);
				*/

				::new(t)shogun::CWeightedDegreeStringKernel(degree, type);
				//t->set_max_mismatch(max_mismatch);

				std::cout << "done loading WDK from non-defaultconstruct data" << std::endl;
			}
	} // serialization
} // namespace boost
#endif //HAVE_BOOST_SERIALIZATION

//BOOST_CLASS_EXPORT_KEY2(shogun::CWeightedDegreeStringKernel, "CWeightedDegreeStringKernel");


#endif /* _WEIGHTEDDEGREESTRINGKERNEL_H__ */
