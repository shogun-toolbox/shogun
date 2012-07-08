/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTLINEARMACHINE_H__
#define __LATENTLINEARMACHINE_H__

#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/labels/LatentLabels.h>
#include <shogun/features/LatentFeatures.h>

namespace shogun
{
	class CLatentLinearMachine;

	typedef void (*argmax_func) (CLatentLinearMachine&, void* userData);
	typedef void (*psi_func) (CLatentLinearMachine&, CLatentData* x, CLatentData* h, float64_t*);
	typedef CLatentData* (*infer_func) (CLatentLinearMachine&, CLatentData* x);

	class CLatentLinearMachine: public CLinearMachine
	{

		public:
			MACHINE_PROBLEM_TYPE(PT_LATENT);

			/** default contstructor */
			CLatentLinearMachine();

			/** constructor
			 *
			 * @param C constant C
			 * @param traindat training features
			 * @param trainlab labels for training features
			 */
			CLatentLinearMachine(float64_t C,
					CLatentFeatures* traindat,
					CLabels* trainlab,
					index_t psi_size);

			virtual ~CLatentLinearMachine();

			/** apply linear machine to all examples
			 *
			 * @return resulting labels
			 */
			virtual CLatentLabels* apply();

			/** apply linear machine to data
			 *
			 * @param data (test)data to be classified
			 * @return classified labels
			 */
			virtual CLatentLabels* apply(CFeatures* data);

			/** get features
			 *
			 * @return features
			 */
			virtual CDotFeatures* get_features()
			{ 
				SG_REF(features);
				return features;
			}

			/** get latent features
			 *
			 * @return features
			 */
			virtual CLatentFeatures* get_latent_features()
			{
				SG_REF(m_latent_feats);
				return m_latent_feats;
			}

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const { return "LatentLinearMachine"; }

			/** set epsilon
			 *
			 * @param eps new epsilon
			 */
			inline void set_epsilon(float64_t eps) { m_epsilon=eps; }

			/** get epsilon
			 *
			 * @return epsilon
			 */
			inline float64_t get_epsilon() { return m_epsilon; }

			/** set C
			 *
			 * @param c_neg new C constant for negatively labeled examples
			 * @param c_pos new C constant for positively labeled examples
			 *
			 * Note that not all SVMs support this (however at least CLibSVM and
			 * CSVMLight do)
			 */
			inline void set_C(float64_t c_neg, float64_t c_pos)
			{
				m_C1=c_neg;
				m_C2=c_pos;
			}

			/** get C1
			 *
			 * @return C1
			 */
			inline float64_t get_C1 () { return m_C1; }

			/** get C2
			 *
			 * @return C2
			 */
			inline float64_t get_C2 () { return m_C2; }

			/** set maximum iterations
			 *
			 * @param iter new maximum iteration value
			 */
			inline void set_max_iterations(int32_t iter) { m_max_iter = iter; }

			/** get maximum iterations value
			 *
			 * @return maximum iterations
			 */
			inline int32_t get_max_iterations() { return m_max_iter; }

			void set_argmax(argmax_func usr_argmax);

			void set_psi(psi_func usr_psi);

			void set_infer(infer_func usr_infer);

			index_t get_psi_size() const { return m_psi_size; }
			void set_psi_size(index_t psi_size) { m_psi_size = psi_size; }

		protected:
			virtual bool train_machine(CFeatures* data=NULL);
			virtual void compute_psi();

		private:
			static void default_argmax_h(CLatentLinearMachine&, void* userData);

			/** initalize the values to default values */
			void init();

		private:
			argmax_func argmax_h;
			psi_func psi;
			infer_func infer;
			/** C1 */
			float64_t m_C1;
			/** C2 */
			float64_t m_C2;
			/** epsilon */
			float64_t m_epsilon;
			/** max iterations */
			int32_t m_max_iter;
			/** psi size */
			index_t m_psi_size;
			/** Latent Features */
			CLatentFeatures* m_latent_feats;
	};
}

#endif /* __LATENTLINEARMACHINE_H__ */

