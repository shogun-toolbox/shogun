/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * 
 */
#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/MatrixOperations.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;
using namespace Eigen;

TEST(MatrixOperations,get_log_det)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance, result;

	index_t size = 5;
	SGMatrix<float64_t> A(size, size);
	A(0,0) = 17.0;
	A(0,1) = 24.0;
	A(0,2) = 1.0;
	A(0,3) = 8.0;
	A(0,4) = 15.0;
	A(1,0) = 23.0;
	A(1,1) = 5.0;
	A(1,2) = 7.0;
	A(1,3) = 14.0;
	A(1,4) = 16.0;
	A(2,0) = 4.0;
	A(2,1) = 6.0;
	A(2,2) = 13.0;
	A(2,3) = 20.0;
	A(2,4) = 22.0;
	A(3,0) = 10.0;
	A(3,1) = 12.0;
	A(3,2) = 19.0;
	A(3,3) = 21.0;
	A(3,4) = 3.0;
	A(4,0) = 11.0;
	A(4,1) = 18.0;
	A(4,2) = 25.0;
	A(4,3) = 2.0;
	A(4,4) = 9.0;
	Map<MatrixXd> eigen_A(A.matrix, A.num_rows, A.num_cols);
	result = CMatrixOperations::get_log_det(eigen_A);
	abs_tolorance = CMath::get_abs_tolorance(15.438851375567365, rel_tolorance);
	EXPECT_NEAR(result, 15.438851375567365, abs_tolorance);

	size = 6;
	SGMatrix<float64_t> B(size, size);
	B(0,0) = 35.000000;
	B(0,1) = 1.000000;
	B(0,2) = 6.000000;
	B(0,3) = 26.000000;
	B(0,4) = 19.000000;
	B(0,5) = 24.000000;
	B(1,0) = 3.000000;
	B(1,1) = 32.000000;
	B(1,2) = 7.000000;
	B(1,3) = 21.000000;
	B(1,4) = 23.000000;
	B(1,5) = 25.000000;
	B(2,0) = 31.000000;
	B(2,1) = 9.000000;
	B(2,2) = 2.000000;
	B(2,3) = 22.000000;
	B(2,4) = 27.000000;
	B(2,5) = 20.000000;
	B(3,0) = 8.000000;
	B(3,1) = 28.000000;
	B(3,2) = 33.000000;
	B(3,3) = 17.000000;
	B(3,4) = 10.000000;
	B(3,5) = 15.000000;
	B(4,0) = 30.000000;
	B(4,1) = 5.000000;
	B(4,2) = 34.000000;
	B(4,3) = 12.000000;
	B(4,4) = 14.000000;
	B(4,5) = 16.000000;
	B(5,0) = 4.000000;
	B(5,1) = 36.000000;
	B(5,2) = 29.000000;
	B(5,3) = 13.000000;
	B(5,4) = 18.000000;
	B(5,5) = 11.000000;
	Map<MatrixXd> eigen_B(B.matrix, B.num_rows, B.num_cols);
	result = CMatrixOperations::get_log_det(eigen_B);
	EXPECT_EQ(result, CMath::INFTY);

	size = 8;
	SGMatrix<float64_t> C(size, size);
	C(0,0) = 64.000000;
	C(0,1) = 2.000000;
	C(0,2) = 3.000000;
	C(0,3) = 61.000000;
	C(0,4) = 60.000000;
	C(0,5) = 6.000000;
	C(0,6) = 7.000000;
	C(0,7) = 57.000000;
	C(1,0) = 9.000000;
	C(1,1) = 55.000000;
	C(1,2) = 54.000000;
	C(1,3) = 12.000000;
	C(1,4) = 13.000000;
	C(1,5) = 51.000000;
	C(1,6) = 50.000000;
	C(1,7) = 16.000000;
	C(2,0) = 17.000000;
	C(2,1) = 47.000000;
	C(2,2) = 46.000000;
	C(2,3) = 20.000000;
	C(2,4) = 21.000000;
	C(2,5) = 43.000000;
	C(2,6) = 42.000000;
	C(2,7) = 24.000000;
	C(3,0) = 40.000000;
	C(3,1) = 26.000000;
	C(3,2) = 27.000000;
	C(3,3) = 37.000000;
	C(3,4) = 36.000000;
	C(3,5) = 30.000000;
	C(3,6) = 31.000000;
	C(3,7) = 33.000000;
	C(4,0) = 32.000000;
	C(4,1) = 34.000000;
	C(4,2) = 35.000000;
	C(4,3) = 29.000000;
	C(4,4) = 28.000000;
	C(4,5) = 38.000000;
	C(4,6) = 39.000000;
	C(4,7) = 25.000000;
	C(5,0) = 41.000000;
	C(5,1) = 23.000000;
	C(5,2) = 22.000000;
	C(5,3) = 44.000000;
	C(5,4) = 45.000000;
	C(5,5) = 19.000000;
	C(5,6) = 18.000000;
	C(5,7) = 48.000000;
	C(6,0) = 49.000000;
	C(6,1) = 15.000000;
	C(6,2) = 14.000000;
	C(6,3) = 52.000000;
	C(6,4) = 53.000000;
	C(6,5) = 11.000000;
	C(6,6) = 10.000000;
	C(6,7) = 56.000000;
	C(7,0) = 8.000000;
	C(7,1) = 58.000000;
	C(7,2) = 59.000000;
	C(7,3) = 5.000000;
	C(7,4) = 4.000000;
	C(7,5) = 62.000000;
	C(7,6) = 63.000000;
	C(7,7) = 1.000000;
	Map<MatrixXd> eigen_C(C.matrix, C.num_rows, C.num_cols);
	result = CMatrixOperations::get_log_det(eigen_C);
	EXPECT_EQ(result, CMath::INFTY);

	size = 13;
	SGMatrix<float64_t> D(size, size);
	D(0,0) = 93.000000;
	D(0,1) = 108.000000;
	D(0,2) = 123.000000;
	D(0,3) = 138.000000;
	D(0,4) = 153.000000;
	D(0,5) = 168.000000;
	D(0,6) = 1.000000;
	D(0,7) = 16.000000;
	D(0,8) = 31.000000;
	D(0,9) = 46.000000;
	D(0,10) = 61.000000;
	D(0,11) = 76.000000;
	D(0,12) = 91.000000;
	D(1,0) = 107.000000;
	D(1,1) = 122.000000;
	D(1,2) = 137.000000;
	D(1,3) = 152.000000;
	D(1,4) = 167.000000;
	D(1,5) = 13.000000;
	D(1,6) = 15.000000;
	D(1,7) = 30.000000;
	D(1,8) = 45.000000;
	D(1,9) = 60.000000;
	D(1,10) = 75.000000;
	D(1,11) = 90.000000;
	D(1,12) = 92.000000;
	D(2,0) = 121.000000;
	D(2,1) = 136.000000;
	D(2,2) = 151.000000;
	D(2,3) = 166.000000;
	D(2,4) = 12.000000;
	D(2,5) = 14.000000;
	D(2,6) = 29.000000;
	D(2,7) = 44.000000;
	D(2,8) = 59.000000;
	D(2,9) = 74.000000;
	D(2,10) = 89.000000;
	D(2,11) = 104.000000;
	D(2,12) = 106.000000;
	D(3,0) = 135.000000;
	D(3,1) = 150.000000;
	D(3,2) = 165.000000;
	D(3,3) = 11.000000;
	D(3,4) = 26.000000;
	D(3,5) = 28.000000;
	D(3,6) = 43.000000;
	D(3,7) = 58.000000;
	D(3,8) = 73.000000;
	D(3,9) = 88.000000;
	D(3,10) = 103.000000;
	D(3,11) = 105.000000;
	D(3,12) = 120.000000;
	D(4,0) = 149.000000;
	D(4,1) = 164.000000;
	D(4,2) = 10.000000;
	D(4,3) = 25.000000;
	D(4,4) = 27.000000;
	D(4,5) = 42.000000;
	D(4,6) = 57.000000;
	D(4,7) = 72.000000;
	D(4,8) = 87.000000;
	D(4,9) = 102.000000;
	D(4,10) = 117.000000;
	D(4,11) = 119.000000;
	D(4,12) = 134.000000;
	D(5,0) = 163.000000;
	D(5,1) = 9.000000;
	D(5,2) = 24.000000;
	D(5,3) = 39.000000;
	D(5,4) = 41.000000;
	D(5,5) = 56.000000;
	D(5,6) = 71.000000;
	D(5,7) = 86.000000;
	D(5,8) = 101.000000;
	D(5,9) = 116.000000;
	D(5,10) = 118.000000;
	D(5,11) = 133.000000;
	D(5,12) = 148.000000;
	D(6,0) = 8.000000;
	D(6,1) = 23.000000;
	D(6,2) = 38.000000;
	D(6,3) = 40.000000;
	D(6,4) = 55.000000;
	D(6,5) = 70.000000;
	D(6,6) = 85.000000;
	D(6,7) = 100.000000;
	D(6,8) = 115.000000;
	D(6,9) = 130.000000;
	D(6,10) = 132.000000;
	D(6,11) = 147.000000;
	D(6,12) = 162.000000;
	D(7,0) = 22.000000;
	D(7,1) = 37.000000;
	D(7,2) = 52.000000;
	D(7,3) = 54.000000;
	D(7,4) = 69.000000;
	D(7,5) = 84.000000;
	D(7,6) = 99.000000;
	D(7,7) = 114.000000;
	D(7,8) = 129.000000;
	D(7,9) = 131.000000;
	D(7,10) = 146.000000;
	D(7,11) = 161.000000;
	D(7,12) = 7.000000;
	D(8,0) = 36.000000;
	D(8,1) = 51.000000;
	D(8,2) = 53.000000;
	D(8,3) = 68.000000;
	D(8,4) = 83.000000;
	D(8,5) = 98.000000;
	D(8,6) = 113.000000;
	D(8,7) = 128.000000;
	D(8,8) = 143.000000;
	D(8,9) = 145.000000;
	D(8,10) = 160.000000;
	D(8,11) = 6.000000;
	D(8,12) = 21.000000;
	D(9,0) = 50.000000;
	D(9,1) = 65.000000;
	D(9,2) = 67.000000;
	D(9,3) = 82.000000;
	D(9,4) = 97.000000;
	D(9,5) = 112.000000;
	D(9,6) = 127.000000;
	D(9,7) = 142.000000;
	D(9,8) = 144.000000;
	D(9,9) = 159.000000;
	D(9,10) = 5.000000;
	D(9,11) = 20.000000;
	D(9,12) = 35.000000;
	D(10,0) = 64.000000;
	D(10,1) = 66.000000;
	D(10,2) = 81.000000;
	D(10,3) = 96.000000;
	D(10,4) = 111.000000;
	D(10,5) = 126.000000;
	D(10,6) = 141.000000;
	D(10,7) = 156.000000;
	D(10,8) = 158.000000;
	D(10,9) = 4.000000;
	D(10,10) = 19.000000;
	D(10,11) = 34.000000;
	D(10,12) = 49.000000;
	D(11,0) = 78.000000;
	D(11,1) = 80.000000;
	D(11,2) = 95.000000;
	D(11,3) = 110.000000;
	D(11,4) = 125.000000;
	D(11,5) = 140.000000;
	D(11,6) = 155.000000;
	D(11,7) = 157.000000;
	D(11,8) = 3.000000;
	D(11,9) = 18.000000;
	D(11,10) = 33.000000;
	D(11,11) = 48.000000;
	D(11,12) = 63.000000;
	D(12,0) = 79.000000;
	D(12,1) = 94.000000;
	D(12,2) = 109.000000;
	D(12,3) = 124.000000;
	D(12,4) = 139.000000;
	D(12,5) = 154.000000;
	D(12,6) = 169.000000;
	D(12,7) = 2.000000;
	D(12,8) = 17.000000;
	D(12,9) = 32.000000;
	D(12,10) = 47.000000;
	D(12,11) = 62.000000;
	D(12,12) = 77.000000;


	Map<MatrixXd> eigen_D(D.matrix, D.num_rows, D.num_cols);
	result = CMatrixOperations::get_log_det(eigen_D);
	abs_tolorance = CMath::get_abs_tolorance(66.001435835567136, rel_tolorance);
	EXPECT_NEAR(result, 66.001435835567136, abs_tolorance);
}
#endif /* HAVE_EIGEN3 */
