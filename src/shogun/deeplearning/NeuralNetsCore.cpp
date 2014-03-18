#include "NeuralNets.h"
#include <iostream>

NeuralNets::NeuralNets()
{
	m_weights = new EigenDenseMat[MAXLAYERS];
	m_bias = new EigenDenseRowVec[MAXLAYERS];
	for (int32_t i = 0; i < NNConfig::layers - 1; ++i)
	{
		m_weights[i] = EigenDenseMat::Random(NNConfig::length[i + 1], NNConfig::length[i]) * 0.1;
		m_bias[i] = EigenDenseRowVec::Zero(NNConfig::length[i + 1]);
		m_db[i] = EigenDenseVec::Zero(NNConfig::length[i + 1]);
		m_vw[i] = EigenDenseMat::Zero(NNConfig::length[i + 1], NNConfig::length[i]);
		m_vb[i] = EigenDenseVec::Zero(NNConfig::length[i + 1]);		
	}
}

float32_t NeuralNets::FeedForward(EigenDenseMat &inputs, const EigenDenseMat& ground_truth)
{
	for (int32_t cur_layer = 1; cur_layer < NNConfig::layers; ++cur_layer)
	{
		//Calulate weighted sum from previous layer's activations
		if (cur_layer == 1)
			m_activations[cur_layer] = inputs * m_weights[cur_layer - 1].transpose();
		else
			m_activations[cur_layer] = m_activations[cur_layer - 1] * m_weights[cur_layer - 1].transpose();

		//Add bias vector
		m_activations[cur_layer] += m_bias[cur_layer - 1].replicate(m_activations[cur_layer].rows(), 1);

		//Apply activation function
		if (cur_layer != NNConfig::layers - 1
			&& NNConfig::opt.act_type != FuncType::LINEAR)
			ApplyActivationFunc(m_activations[cur_layer], NNConfig::opt.act_type);
	}

	//For output layer
	if (NNConfig::opt.out_type != FuncType::LINEAR)
		ApplyActivationFunc(m_activations[NNConfig::layers - 1], NNConfig::opt.out_type);

	return CalcErr(m_activations[NNConfig::layers - 1], ground_truth, m_err);
}

void NeuralNets::BackPropogation(EigenDenseMat &inputs)
{
	//cur_layer ranges from 0 to (layers - 1)
	//weights[cur_layer] connects cur_layer and (cur_layer + 1)
	for (int32_t cur_layer = NNConfig::layers - 2; cur_layer >= 0; --cur_layer)
	{
		if (cur_layer == 0)
			//For weights that connect input layer
			m_dw[cur_layer] = m_err.transpose() * inputs;
		else
			m_dw[cur_layer] = m_err.transpose() * m_activations[cur_layer];

		//Average by the number of samples
		m_dw[cur_layer] /= inputs.rows();
		
		for (int32_t j = 0; j < NNConfig::length[cur_layer + 1]; ++j)
		{
			m_db[cur_layer](j) = 0;
			for (int32_t k = 0; k < m_err.rows(); ++k)
				m_db[cur_layer](j) += m_err(k, j);
			m_db[cur_layer] /= m_err.rows();
		}

		if (cur_layer == 0) break;

		if (NNConfig::opt.act_type != FuncType::LINEAR)
			//After next line, m_activations stores the derivatives of the activation function
			ComputeDerivatives(m_activations[cur_layer], NNConfig::opt.act_type);

		m_err *= m_weights[cur_layer];

		if (NNConfig::opt.act_type != FuncType::LINEAR)
			m_err = m_err.cwiseProduct(m_activations[cur_layer]);
	}
}

void NeuralNets::ApplyGradients()
{
	for (int32_t cur_layer = 0; cur_layer < NNConfig::layers - 1; ++cur_layer)
	{
		if (NNConfig::opt.weightPenaltyL2 > 0)
		{
			m_dw[cur_layer] += m_weights[cur_layer] * NNConfig::opt.weightPenaltyL2;
		}
		m_dw[cur_layer] *= NNConfig::rtc.learning_rate;
		m_vw[cur_layer] *= NNConfig::rtc.momentum;

		m_vw[cur_layer] += m_dw[cur_layer];
		m_weights[cur_layer] -= m_vw[cur_layer];

		for (int32_t j = 0; j < NNConfig::length[cur_layer + 1]; ++j)
		{
			m_db[cur_layer](j) *= NNConfig::rtc.learning_rate;
			m_vb[cur_layer](j) *= NNConfig::rtc.momentum;
			m_vb[cur_layer](j) += m_db[cur_layer](j);
			m_bias[cur_layer](j) -= m_vb[cur_layer](j);
		}
	}
}

float32_t NeuralNets::CalcErr(EigenDenseMat& output, const EigenDenseMat& ground_truth, EigenDenseMat& err)
{
	float32_t loss = 0;
	err = output - ground_truth;

	switch (NNConfig::opt.out_type)
	{
	case FuncType::LINEAR:
		loss = GetSquareLoss(err);
		break;
	case FuncType::SIGM:
		loss = GetSquareLoss(err);
		break;
	case FuncType::SOFTMAX:
		loss = GetLogLoss(output, ground_truth);
	default:
		break;
	}

	return loss;
}
