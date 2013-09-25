/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>
#include <shogun/mathematics/JacobiEllipticFunctions.h>
#include <gtest/gtest.h>

#ifdef HAVE_ARPREC
#include <arprec/mp_real.h>
#include <arprec/mp_complex.h>
#endif //HAVE_ARPREC

using namespace shogun;

typedef float64_t Real;
typedef complex128_t Complex;

TEST(JacobiEllipticFunctions, ellipKKp)
{
	Real K, Kp;
#ifdef HAVE_ARPREC
	CJacobiEllipticFunctions::ellipKKp(0.9999999999, K, Kp);
	EXPECT_NEAR(K, 1.57153044117197637775, 1E-19);
	EXPECT_NEAR(Kp, 4.52953569660335286784, 1E-19);

	CJacobiEllipticFunctions::ellipKKp(10.1, K, Kp);
	EXPECT_NEAR(K, 1.57079632679489655800, 1E-19);
	EXPECT_NEAR(Kp, 33.11638016237679948972, 1E-19);

	CJacobiEllipticFunctions::ellipKKp(10.0, K, Kp);
	EXPECT_NEAR(K, 1.57079632679489655800, 1E-19);
	EXPECT_EQ(Kp, std::numeric_limits<Real>::max());

	CJacobiEllipticFunctions::ellipKKp(0.0, K, Kp);
	EXPECT_EQ(K, std::numeric_limits<Real>::max());
	EXPECT_NEAR(Kp, 1.5707963267948965580, 1E-19);
#else
	CJacobiEllipticFunctions::ellipKKp(0.9999999999, K, Kp);
	EXPECT_NEAR(K, 1.57153044117197637775, 1E-15);
	EXPECT_NEAR(Kp, 4.52953569660335286784, 1E-15);

	CJacobiEllipticFunctions::ellipKKp(10.1, K, Kp);
	EXPECT_NEAR(K, 1.57079632679489655800, 1E-15);
	EXPECT_NEAR(Kp, 33.11638016237679948972, 1E-15);

	CJacobiEllipticFunctions::ellipKKp(10.0, K, Kp);
	EXPECT_NEAR(K, 1.57079632679489655800, 1E-15);
	EXPECT_EQ(Kp, std::numeric_limits<Real>::max());

	CJacobiEllipticFunctions::ellipKKp(0.0, K, Kp);
	EXPECT_EQ(K, std::numeric_limits<Real>::max());
	EXPECT_NEAR(Kp, 1.5707963267948965580, 1E-15);
#endif //HAVE_ARPREC
}

TEST(JacobiEllipticFunctions, ellipJC)
{
	Complex sn, cn, dn;
#ifdef HAVE_ARPREC
	CJacobiEllipticFunctions::ellipJC(Complex(0.5,0.0), 0.0, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.47942553860420300538, 1E-19);
	EXPECT_NEAR(cn.real(), 0.87758256189037275874, 1E-19);
	EXPECT_NEAR(dn.real(), 1.0, 1E-19);
	EXPECT_NEAR(sn.imag(), 0.0, 1E-19);
	EXPECT_NEAR(cn.imag(), 0.0, 1E-19);
	EXPECT_NEAR(dn.imag(), 0.0, 1E-19);
	
	CJacobiEllipticFunctions::ellipJC(Complex(0.5,0.5), 0.0, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.54061268571315335141, 1E-19);
	EXPECT_NEAR(cn.real(), 0.98958488339991990124, 1E-19);
	EXPECT_NEAR(dn.real(), 1.0, 1E-19);
	EXPECT_NEAR(sn.imag(), 0.45730415318424921800, 1E-19);
	EXPECT_NEAR(cn.imag(), -0.24982639750046153893, 1E-19);
	EXPECT_NEAR(dn.imag(), 0.0, 1E-19);
	
	CJacobiEllipticFunctions::ellipJC(Complex(0.5,0.5), 0.5, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.55284325098354925032, 1E-19);
	EXPECT_NEAR(cn.real(), 0.96941944802337476350, 1E-19);
	EXPECT_NEAR(dn.real(), 0.97703467549246159063, 1E-19);
	EXPECT_NEAR(sn.imag(), 0.43032986483620772056, 1E-19);
	EXPECT_NEAR(cn.imag(), -0.24540972636400421036, 1E-19);
	EXPECT_NEAR(dn.imag(), -0.12174847394819815483, 1E-19);
	
	CJacobiEllipticFunctions::ellipJC(Complex(0.2,0.0), 0.99, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.19738823726686019477, 1E-19);
	EXPECT_NEAR(cn.real(), 0.98032539689058428856, 1E-19);
	EXPECT_NEAR(dn.real(), 0.98052409707808552142, 1E-19);
	EXPECT_NEAR(sn.imag(), 0.0, 1E-19);
	EXPECT_NEAR(cn.imag(), 0.0, 1E-19);
	EXPECT_NEAR(dn.imag(), 0.0, 1E-19);

	CJacobiEllipticFunctions::ellipJC(Complex(0.7,0.4), 0.9999999999999999,
		sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.66873789889962576005, 1E-19);
	EXPECT_NEAR(cn.real(), 0.81197156278765458826, 1E-19);
	EXPECT_NEAR(dn.real(), 0.81197156278765469928, 1E-19);
	EXPECT_NEAR(sn.imag(), 0.25191557357137611683, 1E-19);
	EXPECT_NEAR(cn.imag(), -0.20747708305429035658, 1E-19);
	EXPECT_NEAR(dn.imag(), -0.20747708305429032882, 1E-19);
	
	CJacobiEllipticFunctions::ellipJC(Complex(0.5,0.5), 1.0L-1E-16,
		sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.56408314126749847794, 1E-19);
	EXPECT_NEAR(cn.real(), 0.94997886761549465984, 1E-19);
	EXPECT_NEAR(dn.real(), 0.94997886761549465984, 1E-19);
	EXPECT_NEAR(sn.imag(), 0.40389645531602574868, 1E-19);
	EXPECT_NEAR(cn.imag(), -0.23982763093808803778, 1E-19);
	EXPECT_NEAR(dn.imag(), -0.23982763093808801003, 1E-19);
#else
	CJacobiEllipticFunctions::ellipJC(Complex(0.5,0.0), 0.0, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.47942553860420300538, 1E-15);
	EXPECT_NEAR(cn.real(), 0.87758256189037275874, 1E-15);
	EXPECT_NEAR(dn.real(), 1.0, 1E-15);
	EXPECT_NEAR(sn.imag(), 0.0, 1E-15);
	EXPECT_NEAR(cn.imag(), 0.0, 1E-15);
	EXPECT_NEAR(dn.imag(), 0.0, 1E-15);
	
	CJacobiEllipticFunctions::ellipJC(Complex(0.5,0.5), 0.0, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.54061268571315335141, 1E-15);
	EXPECT_NEAR(cn.real(), 0.98958488339991990124, 1E-15);
	EXPECT_NEAR(dn.real(), 1.0, 1E-15);
	EXPECT_NEAR(sn.imag(), 0.45730415318424921800, 1E-15);
	EXPECT_NEAR(cn.imag(), -0.24982639750046153893, 1E-15);
	EXPECT_NEAR(dn.imag(), 0.0, 1E-15);
	
	CJacobiEllipticFunctions::ellipJC(Complex(0.5,0.5), 0.5, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.55284325098354925032, 1E-15);
	EXPECT_NEAR(cn.real(), 0.96941944802337476350, 1E-10);//low precision
	EXPECT_NEAR(dn.real(), 0.97703467549246159063, 1E-15);
	EXPECT_NEAR(sn.imag(), 0.43032986483620772056, 1E-15);
	EXPECT_NEAR(cn.imag(), -0.24540972636400421036, 1E-10);//low precision
	EXPECT_NEAR(dn.imag(), -0.12174847394819815483, 1E-15);
	
	CJacobiEllipticFunctions::ellipJC(Complex(0.2,0.0), 0.99, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.19738823726686019477, 1E-15);
	EXPECT_NEAR(cn.real(), 0.98032539689058428856, 1E-8);//low precision
	EXPECT_NEAR(dn.real(), 0.98052409707808552142, 1E-15);
	EXPECT_NEAR(sn.imag(), 0.0, 1E-15);
	EXPECT_NEAR(cn.imag(), 0.0, 1E-15);
	EXPECT_NEAR(dn.imag(), 0.0, 1E-15);

	CJacobiEllipticFunctions::ellipJC(Complex(0.7,0.4),
		0.9999999999999999, sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.66873789889962576005, 1E-15);
	EXPECT_NEAR(cn.real(), 0.81197156278765458826, 1E-15);
	EXPECT_NEAR(dn.real(), 0.81197156278765469928, 1E-15);
	EXPECT_NEAR(sn.imag(), 0.25191557357137611683, 1E-15);
	EXPECT_NEAR(cn.imag(), -0.20747708305429035658, 1E-15);
	EXPECT_NEAR(dn.imag(), -0.20747708305429032882, 1E-15);
	
	CJacobiEllipticFunctions::ellipJC(Complex(0.5,0.5), 1.0L-1E-16,
		sn, cn, dn);
	EXPECT_NEAR(sn.real(), 0.56408314126749847794, 1E-15);
	EXPECT_NEAR(cn.real(), 0.94997886761549465984, 1E-15);
	EXPECT_NEAR(dn.real(), 0.94997886761549465984, 1E-15);
	EXPECT_NEAR(sn.imag(), 0.40389645531602574868, 1E-15);
	EXPECT_NEAR(cn.imag(), -0.23982763093808803778, 1E-15);
	EXPECT_NEAR(dn.imag(), -0.23982763093808801003, 1E-15);
#endif //HAVE_ARPREC
}
