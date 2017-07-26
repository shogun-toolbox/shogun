/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2017 Giovanni De Toni
*
* This code was adapted from the previous CrossValidationMulticlassStorage
* class, which was written by Heiko Strathmann and Sergey Lisitsyn.
*/

#ifndef SHOGUN_PARAMETEROBSERVERCVMULTICLASS_H
#define SHOGUN_PARAMETEROBSERVERCVMULTICLASS_H

#include <shogun/lib/parameter_observers/ParameterObserverCV.h>

namespace shogun
{

	class CMulticlassLabels;
	class CDynamicObjectArray;
	class CBinaryClassEvaluation;

	/**
	 * Observer which store values and generate stats about
	 * a multiclass cross validation.
	 */
	class ParameterObserverCVMulticlass : public ParameterObserverCV
	{

	public:
		/**
		 * Constructor
		 * @param compute_ROC whether we want to compute ROC
		 * @param compute_PRC whether we want to compute PRC
		 * @param compute_conf_matrices whether we want to compute confidence
		 * matrices
		 */
		ParameterObserverCVMulticlass(
		    bool compute_ROC = true, bool compute_PRC = false,
		    bool compute_conf_matrices = false, bool verbose = false);

		/**
		 * Destructor
		 */
		virtual ~ParameterObserverCVMulticlass();

		/** Appends a binary evaluation instance
		*
		* @param evaluation binary evaluation to add
		*/
		void append_binary_evaluation(CBinaryClassEvaluation* evaluation);

		/** Returns binary evaluation appended before
		 *
		 * @param idx
		 */
		CBinaryClassEvaluation* get_binary_evaluation(int32_t idx);

		/** Returns ROC of 1-v-R in given fold and run
		*
		* @param run run
		* @param fold fold
		* @param c class
		* @return ROC of 'run' run, 'fold' fold and 'c' class
		*/
		SGMatrix<float64_t> get_fold_ROC(int32_t run, int32_t fold, int32_t c);

		/** Returns PRC of 1-v-R in given fold and run
		*
		* @param run run
		* @param fold fold
		* @param c class
		* @return ROC of 'run' run, 'fold' fold and 'c' class
		*/
		SGMatrix<float64_t> get_fold_PRC(int32_t run, int32_t fold, int32_t c);

		/** Returns evaluation result of 1-v-R in given fold and run
		*
		* @param run run
		* @param fold fold
		* @param c class
		* @param e evaluation number
		*/
		float64_t get_fold_evaluation_result(
		    int32_t run, int32_t fold, int32_t c, int32_t e);

		/** Returns accuracy of fold and run
		 * @param run run
		 * @param fold fold
		 */
		float64_t get_fold_accuracy(int32_t run, int32_t fold);

		/** Returns confusion matrix of fold and run
		 * @param run run
		 * @param fold fold
		 */
		SGMatrix<int32_t> get_fold_conf_matrix(int32_t run, int32_t fold);

	protected:
		/**
		 * Initialize all the data structure (it must be called only once).
		 * @param name name of the method which will call this initialize (error
		 * reporting purpose)
		 */
		virtual void initialize(std::string name);

		/**
		 * Compute ROC/PRC for a specific run and a specific fold
		 * @param storage storage object which contains a run
		 * @param fold fold object which contains fold's data
		 */
		virtual void compute(
		    CrossValidationStorage* storage, CrossValidationFoldStorage* fold);

		/** if the data structure are initialized*/
		bool m_initialized;

		/** if we want to compute ROC */
		bool m_compute_ROC;

		/** if we want to compute PRC */
		bool m_compute_PRC;

		/** if we want to compute confidence matrixes*/
		bool m_compute_conf_matrices;

		/** fold ROC graphs */
		SGMatrix<float64_t>* m_fold_ROC_graphs;

		/** fold PRC graphs */
		SGMatrix<float64_t>* m_fold_PRC_graphs;

		/** confusion matrices */
		SGMatrix<int32_t>* m_conf_matrices;

		/** number of classes */
		int32_t m_num_classes;

		/** number of runs */
		int64_t m_num_runs;

		/** number of folds */
		int64_t m_num_folds;

		/** predicted results */
		CMulticlassLabels* m_pred_labels;

		/** true labels */
		CMulticlassLabels* m_true_labels;

		/** accuracies */
		SGVector<float64_t> m_accuracies;

		/** custom binary evaluators */
		CDynamicObjectArray* m_binary_evaluations;

		/** fold evaluation results */
		SGVector<float64_t> m_evaluations_results;
	};
}

#endif // SHOGUN_PARAMETEROBSERVERCVMULTICLASS_H
