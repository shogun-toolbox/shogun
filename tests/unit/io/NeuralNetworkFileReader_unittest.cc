/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2014 Khaled Nasr
 */

#include <shogun/lib/config.h>

#ifdef HAVE_JSON

#include <shogun/io/NeuralNetworkFileReader.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/lib/SGVector.h>

#include <gtest/gtest.h>

using namespace shogun;

TEST(NeuralNetworkFileReader, read)
{
	const char* net_string =
	"{"
	"	\"sigma\": 0.01,"
	"	\"optimization_method\": \"NNOM_GRADIENT_DESCENT\","
	"	\"l2_coefficient\": 0.001,"
	"	\"l1_coefficient\": 0.003,"
	"	\"dropout_hidden\": 0.5,"
	"	\"dropout_input\": 0.2,"
	"	\"max_norm\": 15,"
	"	\"epsilon\": 1e-8,"
	"	\"max_num_epochs\": 1000,"
	"	\"gd_mini_batch_size\": 100,"
	"	\"gd_learning_rate\": 1.0,"
	"	\"gd_learning_rate_decay\": 0.995,"
	"	\"gd_momentum\": 0.95,"
	"	\"gd_error_damping_coeff\": 0.9,"

	"	\"layers\":"
	"	{"
	"		\"input1\":"
	"		{"
	"			\"type\": \"NeuralInputLayer\","
	"			\"num_neurons\": 6,"
	"			\"start_index\": 0"
	"		},"
	"		\"input2\":"
	"		{"
	"			\"type\": \"NeuralInputLayer\","
	"			\"num_neurons\": 10,"
	"			\"start_index\": 6"
	"		},"
	"		\"logistic1\":"
	"		{"
	"			\"type\": \"NeuralLogisticLayer\","
	"			\"num_neurons\": 32,"
	"			\"inputs\": [\"input1\", \"input2\"]"
	"		},"
	"		\"linear1\":"
	"		{"
	"			\"type\": \"NeuralLinearLayer\","
	"			\"num_neurons\": 8,"
	"			\"inputs\": [\"logistic1\"]"
	"		},"
	"		\"rectified1\":"
	"		{"
	"			\"type\": \"NeuralRectifiedLinearLayer\","
	"			\"num_neurons\": 8,"
	"			\"inputs\": [\"logistic1\", \"input2\"]"
	"		},"
	"		\"softmax\":"
	"		{"
	"			\"type\": \"NeuralSoftmaxLayer\","
	"			\"num_neurons\": 4,"
	"			\"inputs\": [\"linear1\", \"rectified1\"]"
	"		}"
	"	}"
	"}";

	CNeuralNetworkFileReader reader;
	CNeuralNetwork* net = reader.read_string(net_string);

	EXPECT_EQ(NNOM_GRADIENT_DESCENT, net->optimization_method);
	EXPECT_EQ(0.001, net->l2_coefficient);
	EXPECT_EQ(0.003, net->l1_coefficient);
	EXPECT_EQ(0.5, net->dropout_hidden);
	EXPECT_EQ(0.2, net->dropout_input);
	EXPECT_EQ(15, net->max_norm);
	EXPECT_EQ(1e-8, net->epsilon);
	EXPECT_EQ(1000, net->max_num_epochs);
	EXPECT_EQ(100, net->gd_mini_batch_size);
	EXPECT_EQ(1.0, net->gd_learning_rate);
	EXPECT_EQ(0.995, net->gd_learning_rate_decay);
	EXPECT_EQ(0.95, net->gd_momentum);
	EXPECT_EQ(0.9, net->gd_error_damping_coeff);

	CDynamicObjectArray* layers = net->get_layers();

	CNeuralLayer* input1 = (CNeuralLayer*)layers->element(0);
	EXPECT_EQ(0, strcmp(input1->get_name(), "NeuralInputLayer"));
	EXPECT_EQ(6, input1->get_num_neurons());
	EXPECT_EQ(0, ((CNeuralInputLayer*)input1)->get_start_index());
	SG_UNREF(input1);

	CNeuralLayer* input2 = (CNeuralLayer*)layers->element(1);
	EXPECT_EQ(0, strcmp(input1->get_name(), "NeuralInputLayer"));
	EXPECT_EQ(10, input2->get_num_neurons());
	EXPECT_EQ(6, ((CNeuralInputLayer*)input2)->get_start_index());
	SG_UNREF(input2);

	CNeuralLayer* logistic1 = (CNeuralLayer*)layers->element(2);
	EXPECT_EQ(0, strcmp(logistic1->get_name(), "NeuralLogisticLayer"));
	EXPECT_EQ(32, logistic1->get_num_neurons());
	EXPECT_EQ(2, logistic1->get_input_indices().vlen);
	EXPECT_EQ(0, logistic1->get_input_indices()[0]);
	EXPECT_EQ(1, logistic1->get_input_indices()[1]);
	SG_UNREF(logistic1);

	CNeuralLayer* linear1 = (CNeuralLayer*)layers->element(3);
	EXPECT_EQ(0, strcmp(linear1->get_name(), "NeuralLinearLayer"));
	EXPECT_EQ(8, linear1->get_num_neurons());
	EXPECT_EQ(1, linear1->get_input_indices().vlen);
	EXPECT_EQ(2, linear1->get_input_indices()[0]);
	SG_UNREF(linear1);

	CNeuralLayer* rectified1 = (CNeuralLayer*)layers->element(4);
	EXPECT_EQ(0, strcmp(rectified1->get_name(), "NeuralRectifiedLinearLayer"));
	EXPECT_EQ(8, rectified1->get_num_neurons());
	EXPECT_EQ(2, rectified1->get_input_indices().vlen);
	EXPECT_EQ(1, rectified1->get_input_indices()[0]);
	EXPECT_EQ(2, rectified1->get_input_indices()[1]);
	SG_UNREF(rectified1);

	CNeuralLayer* softmax = (CNeuralLayer*)layers->element(5);
	EXPECT_EQ(0, strcmp(softmax->get_name(), "NeuralSoftmaxLayer"));
	EXPECT_EQ(4, softmax->get_num_neurons());
	EXPECT_EQ(2, softmax->get_input_indices().vlen);
	EXPECT_EQ(3, softmax->get_input_indices()[0]);
	EXPECT_EQ(4, softmax->get_input_indices()[1]);
	SG_UNREF(softmax);

	SG_UNREF(layers);
	SG_UNREF(net);
}

#endif
