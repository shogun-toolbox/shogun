#include "NeuralNets.h"
#include <iostream>

CNeuralNets::CNeuralNets()
{
	m_weights = new EigenDenseMat[MAXLAYERS];
	m_bias = new EigenDenseRowVec[MAXLAYERS];
	for (int32_t i = 0; i < CNNConfig::layers - 1; ++i)
	{
		m_weights[i] = EigenDenseMat::Random(CNNConfig::length[i + 1], CNNConfig::length[i]) * 0.1;
		m_bias[i] = EigenDenseRowVec::Zero(CNNConfig::length[i + 1]);
		m_db[i] = EigenDenseVec::Zero(CNNConfig::length[i + 1]);
		m_vw[i] = EigenDenseMat::Zero(CNNConfig::length[i + 1], CNNConfig::length[i]);
		m_vb[i] = EigenDenseVec::Zero(CNNConfig::length[i + 1]);		
	}
}

float32_t CNeuralNets::FeedForward(EigenDenseMat &inputs, const EigenDenseMat& ground_truth)
{
	for (int32_t cur_layer = 1; cur_layer < CNNConfig::layers; ++cur_layer)
	{
		//Calulate weighted sum from previous layer's activations
		if (cur_layer == 1)
			m_activations[cur_layer] = inputs * m_weights[cur_layer - 1].transpose();
		else
			m_activations[cur_layer] = m_activations[cur_layer - 1] * m_weights[cur_layer - 1].transpose();

		//Add bias vector
		m_activations[cur_layer] += m_bias[cur_layer - 1].replicate(m_activations[cur_layer].rows(), 1);

		//Apply activation function
		if (cur_layer != CNNConfig::layers - 1
			&& CNNConfig::opt.act_type != CFuncType::LINEAR)
			ApplyActivationFunc(m_activations[cur_layer], CNNConfig::opt.act_type);
	}

	//For output layer
	if (CNNConfig::opt.out_type != CFuncType::LINEAR)
		ApplyActivationFunc(m_activations[CNNConfig::layers - 1], CNNConfig::opt.out_type);

	return CalcErr(m_activations[CNNConfig::layers - 1], ground_truth, m_err);
}

void CNeuralNets::BackPropogation(EigenDenseMat &inputs)
{
	//cur_layer ranges from 0 to (layers - 1)
	//weights[cur_layer] connects cur_layer and (cur_layer + 1)
	for (int32_t cur_layer = CNNConfig::layers - 2; cur_layer >= 0; --cur_layer)
	{
		if (cur_layer == 0)
			//For weights that connect input layer
			m_dw[cur_layer] = m_err.transpose() * inputs;
		else
			m_dw[cur_layer] = m_err.transpose() * m_activations[cur_layer];

		//Average by the number of samples
		m_dw[cur_layer] /= inputs.rows();
		
		for (int32_t j = 0; j < CNNConfig::length[cur_layer + 1]; ++j)
		{
			m_db[cur_layer](j) = 0;
			for (int32_t k = 0; k < m_err.rows(); ++k)
				m_db[cur_layer](j) += m_err(k, j);
			m_db[cur_layer] /= m_err.rows();
		}

		if (cur_layer == 0) break;

		if (CNNConfig::opt.act_type != CFuncType::LINEAR)
			//After next line, m_activations stores the derivatives of the activation function
			ComputeDerivatives(m_activations[cur_layer], CNNConfig::opt.act_type);

		m_err *= m_weights[cur_layer];

		if (CNNConfig::opt.act_type != CFuncType::LINEAR)
			m_err = m_err.cwiseProduct(m_activations[cur_layer]);
	}
}

void CNeuralNets::ApplyGradients()
{
	for (int32_t cur_layer = 0; cur_layer < CNNConfig::layers - 1; ++cur_layer)
	{
		if (CNNConfig::opt.weightPenaltyL2 > 0)
		{
			m_dw[cur_layer] += m_weights[cur_layer] * CNNConfig::opt.weightPenaltyL2;
		}
		m_dw[cur_layer] *= CNNConfig::rtc.learning_rate;
		m_vw[cur_layer] *= CNNConfig::rtc.momentum;

		m_vw[cur_layer] += m_dw[cur_layer];
		m_weights[cur_layer] -= m_vw[cur_layer];

		for (int32_t j = 0; j < CNNConfig::length[cur_layer + 1]; ++j)
		{
			m_db[cur_layer](j) *= CNNConfig::rtc.learning_rate;
			m_vb[cur_layer](j) *= CNNConfig::rtc.momentum;
			m_vb[cur_layer](j) += m_db[cur_layer](j);
			m_bias[cur_layer](j) -= m_vb[cur_layer](j);
		}
	}
}

float32_t CNeuralNets::CalcErr(EigenDenseMat& output, const EigenDenseMat& ground_truth, EigenDenseMat& err)
{
	float32_t loss = 0;
	err = output - ground_truth;

	switch (CNNConfig::opt.out_type)
	{
	case CFuncType::LINEAR:
		loss = GetSquareLoss(err);
		break;
	case CFuncType::SIGM:
		loss = GetSquareLoss(err);
		break;
	case CFuncType::SOFTMAX:
		loss = GetLogLoss(output, ground_truth);
	default:
		break;
	}

	return loss;
}
