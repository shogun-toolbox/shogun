// Distributed under GNU General Public License (see license.txt for details).
//
//  Copyright (c) 2007 Shai Shalev-Shwartz.
//  All Rights Reserved.

#include <shogun/optimization/pegasos/pegasos_optimize.h>

using namespace shogun;

// ------------------------------------------------------------//
// ---------------- OPTIMIZING --------------------------------//
// ------------------------------------------------------------//
SGVector<float64_t> CPegasos::Learn(// Input variables
		CDotFeatures* features,
		SGVector<float64_t> labels,
		int dimension,
		double lambda,int max_iter,int exam_per_iter,int num_iter_to_avg,
		// Output variables
		double& obj_value, double& norm_value,double& loss_value,
		// additional parameters
		int eta_rule_type , double eta_constant ,
		int projection_rule, double projection_constant) 
{

	int num_examples = features->get_num_vectors();

	// Initialization of classification vector
	SGVector<float64_t> W(dimension);
	SGVector<float64_t> AvgW(dimension);
	double avgScale = (num_iter_to_avg > max_iter) ? max_iter : num_iter_to_avg; 

	// ---------------- Main Loop -------------------
	for (int i = 0; i < max_iter; ++i) {

		// learning rate
		double eta;
		if (eta_rule_type == 0) { // Pegasos eta rule
			eta = 1 / (lambda * (i+2)); 
		} else if (eta_rule_type == 1) { // Norma rule
			eta = eta_constant / sqrt(i+2);
		} else {
			eta = eta_constant;
		}

		// gradient indices and losses
		std::vector<uint> grad_index;
		std::vector<double> grad_weights;

		// calc sub-gradients
		for (int j=0; j < exam_per_iter; ++j) {

			// choose random example
			uint r = ((int)rand()) % num_examples;

			// calculate prediction
			double prediction = features->dense_dot_sgvec(r, W);

			// calculate loss
			double cur_loss = 1 - labels[r]*prediction;
			if (cur_loss < 0.0) cur_loss = 0.0;

			// and add to the gradient
			if (cur_loss > 0.0) {
				grad_index.push_back(r);
				grad_weights.push_back(eta*labels[r]/exam_per_iter);
			}
		}

		// scale w 
		// W.scale(1.0 - eta*lambda);
		double scaling = 1.0 - eta*lambda;
		if (scaling==0)
			W.zero();
		else 
		{
			SGVector<float64_t>::scale_vector(scaling, W.vector, W.vlen);
		}

		// and add sub-gradients
		for (uint j=0; j<grad_index.size(); ++j) {
			//W.add(Dataset[grad_index[j]],grad_weights[j]);
			features->add_to_dense_vec(grad_weights[j],grad_index[j],W.vector,W.vlen);
		}

		// Project if needed
		if (projection_rule == 0) { // Pegasos projection rule
			double norm2 = SGVector<float64_t>::twonorm(W.vector, W.vlen);
			if (norm2 > 1.0/lambda) {
				SGVector<float64_t>::scale_vector(sqrt(1.0/(lambda*norm2)), W.vector, W.vlen);
			}
		} else if (projection_rule == 1) { // other projection
			double norm2 = SGVector<float64_t>::twonorm(W.vector, W.vlen);
			if (norm2 > (projection_constant*projection_constant)) {
				SGVector<float64_t>::scale_vector(projection_constant/sqrt(norm2), W.vector, W.vlen);
			}
		} // else -- no projection


		// and update AvgW
		if (max_iter <= num_iter_to_avg + i)
			for (int j=0; j<dimension; j++)
				AvgW[j] += W[j]/avgScale;
	}

	// Calculate objective value
	norm_value = SGVector<float64_t>::twonorm(AvgW.vector, AvgW.vlen);
	obj_value = norm_value * lambda / 2.0;
	loss_value = 0.0;
	for (int i=0; i < num_examples; ++i) {
		double cur_loss = 1 - labels[i]*features->dense_dot_sgvec(i,AvgW); 
		if (cur_loss < 0.0) cur_loss = 0.0;
		loss_value += cur_loss/num_examples;
		obj_value += cur_loss/num_examples;
	}
	return AvgW;
}

