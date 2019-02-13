/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser, Chiyuan Zhang, Yuyu Zhang, 
 *          Sergey Lisitsyn
 */

#if !defined(_MUNKRES_H_)
#define _MUNKRES_H_

#include <shogun/lib/config.h>

#include <shogun/lib/DataType.h>
#include <shogun/lib/SGMatrix.h>

#include <list>
#include <utility>

namespace shogun
{

/** @brief Munkres */
class SHOGUN_EXPORT Munkres
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
