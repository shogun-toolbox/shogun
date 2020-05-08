/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

extern "C" {
#include <shogun/mathematics/zbsubs.h>
}

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

#include <shogun/io/SGIO.h>

namespace shogun
{
	double cyl_bessel_k(double nu, double x)
	{
#ifdef _LIBCPP_VERSION
		double zr = x;
		double zi = 0;
		int kode = 1;
		int n = 1;
		double cy[2] = {std::numeric_limits<double>::quiet_NaN(),
		                std::numeric_limits<double>::quiet_NaN()};
		int nz = 1;
		int ierr = 0;

		zbesk_(&zr, &zi, &nu, &kode, &n, &cy[0], &cy[1], &nz, &ierr);

		if (ierr)
		{
			if (ierr == 3)
			{
				io::warn(
				    "Large arguments in cyl_bessel_k (nu={}, x={}) caused "
				    "precision loss of at least half machine accuracy",
				    nu, x);
			}
			else if (ierr == 2)
			{
				io::warn("Overflow in cyl_bessel_k call (nu={}, x={})", nu, x);
				cy[0] = std::numeric_limits<double>::infinity();
			}
			else if (ierr == 4)
			{
				io::warn(
				    "|x|={} or nu={} too large for cyl_bessel_k", std::abs(x),
				    nu);
				cy[0] = std::numeric_limits<double>::quiet_NaN();
			}
			else
			{
				error("Unexpected error {} in cyl_bessel_k", ierr);
			}
		}
		return cy[0];
#else
		return std::cyl_bessel_k(nu, x);
#endif
	}
}
