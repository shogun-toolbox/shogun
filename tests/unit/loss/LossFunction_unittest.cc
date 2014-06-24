/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/loss/LossFunction.h>
#include <shogun/loss/ExponentialLoss.h>
#include <shogun/loss/AbsoluteDeviationLoss.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/loss/HuberLoss.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

void set_values(SGVector<float64_t> predicted, SGVector<float64_t> actual)
{
	predicted[0]=-1.61606154325792106;
	predicted[1]=1.00759733375338723;
	predicted[2]=3.50337445933915959;
	predicted[3]=4.58654355713784412;
	predicted[4]=-2.44021069431682402;

	actual[0]=0.56734951808288514;
	actual[1]=4.94796903337071825;
	actual[2]=-1.29006464646432173;
	actual[3]=-3.07539812244367727;
	actual[4]=2.910904914119957;
}

TEST(LossFunction, squared_loss_test)
{
	SGVector<float64_t> predicted(5);
	SGVector<float64_t> actual(5);
	set_values(predicted,actual);

	CLossFunction* lossf=new CSquaredLoss();

	SGVector<float64_t> loss(5);
	SGVector<float64_t> firstd(5);
	SGVector<float64_t> secondd(5);

	for (int32_t i=0;i<5;i++)
	{
		loss[i]=lossf->loss(predicted[i],actual[i]);
		firstd[i]=lossf->first_derivative(predicted[i],actual[i]);
		secondd[i]=lossf->second_derivative(predicted[i],actual[i]);
	}

	float64_t epsilon=1e-7;
	EXPECT_NEAR(loss[0],4.767283862,epsilon);
	EXPECT_NEAR(loss[1],15.526529131,epsilon);
	EXPECT_NEAR(loss[2],22.977058461,epsilon);
	EXPECT_NEAR(loss[3],58.705350301,epsilon);
	EXPECT_NEAR(loss[4],28.634438254,epsilon);

	EXPECT_NEAR(firstd[0],-4.366822122,epsilon);
	EXPECT_NEAR(firstd[1],-7.880743399,epsilon);
	EXPECT_NEAR(firstd[2],9.586878211,epsilon);
	EXPECT_NEAR(firstd[3],15.323883359,epsilon);
	EXPECT_NEAR(firstd[4],-10.702231216,epsilon);

	EXPECT_NEAR(secondd[0],2,epsilon);
	EXPECT_NEAR(secondd[1],2,epsilon);
	EXPECT_NEAR(secondd[2],2,epsilon);
	EXPECT_NEAR(secondd[3],2,epsilon);
	EXPECT_NEAR(secondd[4],2,epsilon);

	SG_UNREF(lossf);
}

TEST(LossFunction, exponential_loss_test)
{
	SGVector<float64_t> predicted(5);
	SGVector<float64_t> actual(5);
	set_values(predicted,actual);

	CLossFunction* lossf=new CExponentialLoss();

	SGVector<float64_t> loss(5);
	SGVector<float64_t> firstd(5);
	SGVector<float64_t> secondd(5);

	for (int32_t i=0;i<5;i++)
	{
		loss[i]=lossf->loss(predicted[i],actual[i]);
		firstd[i]=lossf->first_derivative(predicted[i],actual[i]);
		secondd[i]=lossf->second_derivative(predicted[i],actual[i]);
	}

	float64_t epsilon=1e-7;
	EXPECT_NEAR(loss[0],2.501452936,epsilon);
	EXPECT_NEAR(loss[1],0.006835946,epsilon);
	EXPECT_NEAR(loss[2],91.796992285,epsilon);
	EXPECT_NEAR(loss[3],1336343.143621304,epsilon);
	EXPECT_NEAR(loss[4],1215.877480856,epsilon);

	EXPECT_NEAR(firstd[0],-1.419198118,epsilon);
	EXPECT_NEAR(firstd[1],-0.033824049,epsilon);
	EXPECT_NEAR(firstd[2],118.424054399,epsilon);
	EXPECT_NEAR(firstd[3],4109787.194833442,epsilon);
	EXPECT_NEAR(firstd[4],-3539.303733991,epsilon);

	EXPECT_NEAR(secondd[0],0.805181368,epsilon);
	EXPECT_NEAR(secondd[1],0.167360348,epsilon);
	EXPECT_NEAR(secondd[2],152.774685872,epsilon);
	EXPECT_NEAR(secondd[3],12639231.822633834,epsilon);
	EXPECT_NEAR(secondd[4],10302.576631839,epsilon);

	SG_UNREF(lossf);
}

TEST(LossFunction, abs_deviation_loss_test)
{
	SGVector<float64_t> predicted(5);
	SGVector<float64_t> actual(5);
	set_values(predicted,actual);

	CLossFunction* lossf=new CAbsoluteDeviationLoss();

	SGVector<float64_t> loss(5);
	SGVector<float64_t> firstd(5);
	SGVector<float64_t> secondd(5);

	for (int32_t i=0;i<5;i++)
	{
		loss[i]=lossf->loss(predicted[i],actual[i]);
		firstd[i]=lossf->first_derivative(predicted[i],actual[i]);
		secondd[i]=lossf->second_derivative(predicted[i],actual[i]);
	}

	float64_t epsilon=1e-7;
	EXPECT_NEAR(loss[0],2.183411061,epsilon);
	EXPECT_NEAR(loss[1],3.940371699,epsilon);
	EXPECT_NEAR(loss[2],4.793439105,epsilon);
	EXPECT_NEAR(loss[3],7.661941679,epsilon);
	EXPECT_NEAR(loss[4],5.351115608,epsilon);

	EXPECT_NEAR(firstd[0],-1,epsilon);
	EXPECT_NEAR(firstd[1],-1,epsilon);
	EXPECT_NEAR(firstd[2],1,epsilon);
	EXPECT_NEAR(firstd[3],1,epsilon);
	EXPECT_NEAR(firstd[4],-1,epsilon);

	EXPECT_NEAR(secondd[0],0,epsilon);
	EXPECT_NEAR(secondd[1],0,epsilon);
	EXPECT_NEAR(secondd[2],0,epsilon);
	EXPECT_NEAR(secondd[3],0,epsilon);
	EXPECT_NEAR(secondd[4],0,epsilon);

	SG_UNREF(lossf);
}

TEST(LossFunction, huber_loss_test)
{
	SGVector<float64_t> predicted(5);
	SGVector<float64_t> actual(5);
	set_values(predicted,actual);

	CLossFunction* lossf=new CHuberLoss(4);

	SGVector<float64_t> loss(5);
	SGVector<float64_t> firstd(5);
	SGVector<float64_t> secondd(5);

	for (int32_t i=0;i<5;i++)
	{
		loss[i]=lossf->loss(predicted[i],actual[i]);
		firstd[i]=lossf->first_derivative(predicted[i],actual[i]);
		secondd[i]=lossf->second_derivative(predicted[i],actual[i]);
	}

	float64_t epsilon=1e-7;
	EXPECT_NEAR(loss[0],4.767283862,epsilon);
	EXPECT_NEAR(loss[1],15.526529131,epsilon);
	EXPECT_NEAR(loss[2],11.173756423,epsilon);
	EXPECT_NEAR(loss[3],22.647766718,epsilon);
	EXPECT_NEAR(loss[4],13.404462433,epsilon);

	EXPECT_NEAR(firstd[0],-4.366822122,epsilon);
	EXPECT_NEAR(firstd[1],-7.880743399,epsilon);
	EXPECT_NEAR(firstd[2],4,epsilon);
	EXPECT_NEAR(firstd[3],4,epsilon);
	EXPECT_NEAR(firstd[4],-4,epsilon);

	EXPECT_NEAR(secondd[0],2,epsilon);
	EXPECT_NEAR(secondd[1],2,epsilon);
	EXPECT_NEAR(secondd[2],0,epsilon);
	EXPECT_NEAR(secondd[3],0,epsilon);
	EXPECT_NEAR(secondd[4],0,epsilon);

	SG_UNREF(lossf);
}
