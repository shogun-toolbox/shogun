/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <lib/common.h>
#include <lib/SGVector.h>
#include <mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(CMath, complex_test)
{
	complex128_t a(5.0, 6.0), result;

	EXPECT_NEAR(CMath::abs(a), 7.81024967590665439412, 1E-14);
	result=CMath::log(a);
	EXPECT_NEAR(result.real(), 2.05543693208665567695, 1E-14);
	EXPECT_NEAR(result.imag(), 0.87605805059819341629, 1E-14);
	result=CMath::log10(a);
	EXPECT_NEAR(result.real(), 0.89266491750538345951, 1E-14);
	EXPECT_NEAR(result.imag(), 0.38046717720171513433, 1E-14);
	result=CMath::exp(a);
	EXPECT_NEAR(result.real(), 142.50190551820736573063, 1E-13);
	EXPECT_NEAR(result.imag(), -41.46893678992289267171, 1E-13);

	result=CMath::sqrt(a);
	EXPECT_NEAR(result.real(), 2.53083481048315883655, 1E-14);
	EXPECT_NEAR(result.imag(), 1.18537961765559618499, 1E-14);
	result=CMath::pow(a, 0.25);
	EXPECT_NEAR(result.real(), 1.63179612745502011784, 1E-14);
	EXPECT_NEAR(result.imag(), 0.36321314829455331186, 1E-14);
	result=CMath::pow(1.5, a);
	EXPECT_NEAR(result.real(), -5.76473627294186652392, 1E-14);
	EXPECT_NEAR(result.imag(), 4.94296012182258248657, 1E-14);
	result=CMath::pow(a, a/10.0);
	EXPECT_NEAR(result.real(), -0.16575427581944451871, 1E-14);
	EXPECT_NEAR(result.imag(), 1.64382444391412629869, 1E-14);

	result=CMath::sin(a);
	EXPECT_NEAR(result.real(), -193.43002005693958267329, 1E-14);
	EXPECT_NEAR(result.imag(), 57.21839505634108746790, 1E-14);
	result=CMath::sinh(a);
	EXPECT_NEAR(result.real(), 71.24771797085288937978, 1E-13);
	EXPECT_NEAR(result.imag(), -20.73540973837024026238, 1E-14);

	result=CMath::cos(a);
	EXPECT_NEAR(result.real(), 57.21909818460073893220, 1E-13);
	EXPECT_NEAR(result.imag(), 193.42764312130648818311, 1E-13);
	result=CMath::cosh(a);
	EXPECT_NEAR(result.real(), 71.25418754735444792914, 1E-13);
	EXPECT_NEAR(result.imag(), -20.73352705155264885661, 1E-14);

	result=CMath::tan(a);
	EXPECT_NEAR(result.real(), -0.00000668523139027702, 1E-14);
	EXPECT_NEAR(result.imag(), 1.00001031089811975860, 1E-14);
	result=CMath::tanh(a);
	EXPECT_NEAR(result.real(), 0.99992337992770763400, 1E-14);
	EXPECT_NEAR(result.imag(), -0.00004871701269270740, 1E-14);

	SGVector<complex128_t> vec(10);
	for (index_t i=0; i<10; ++i)
	{
		vec[i]=i%2==0? a : complex128_t(0.0);
	}
	EXPECT_EQ(CMath::get_num_nonzero(vec.vector, 10), 5);
}
