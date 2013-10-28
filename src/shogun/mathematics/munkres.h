/*
 *   Copyright (c) 2007 John Weaver
 *
 *   2012: Ported to shogun by Chiyuan Zhang
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License along
 *   with this program; if not, write to the Free Software Foundation, Inc.,
 *   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#if !defined(_MUNKRES_H_)
#define _MUNKRES_H_

#include <shogun/lib/DataType.h>
#include <shogun/lib/SGMatrix.h>

#include <list>
#include <utility>

namespace shogun
{

/** @brief Munkres */
class Munkres
{
public:
	/** constructor */
	Munkres(SGMatrix<double> &m)
		:mask_matrix(m.num_rows, m.num_cols, true), matrix(m.num_rows, m.num_cols, true), ref_m(m)
	{
	}

	/** solve  */
	void solve()
	{
		solve(ref_m);
	}

	/** destructor */
	~Munkres()
	{
	}

private:
	static const int NORMAL=0;
	static const int STAR=1;
	static const int PRIME=2;

	void solve(SGMatrix<double> &m);

	inline bool find_uncovered_in_matrix(double,int&,int&);
	inline bool pair_in_list(const std::pair<int,int> &, const std::list<std::pair<int,int> > &);
	int step1(void);
	int step2(void);
	int step3(void);
	int step4(void);
	int step5(void);
	int step6(void);
	SGMatrix<int> mask_matrix;
	SGMatrix<double> matrix;
	bool *row_mask;
	bool *col_mask;
	int saverow, savecol;

	SGMatrix<double> &ref_m;
};

} // namespace shogun

#endif /* !defined(_MUNKRES_H_) */
