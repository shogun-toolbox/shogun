#include <shogun/io/SGIO.h>
#include <shogun/base/init.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/structure/FactorGraph.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/Factor.h>
#include <shogun/structure/FactorGraphLabels.h>

#include <iostream>
using namespace std;
using namespace shogun;

inline int grid_to_index(int32_t x, int32_t y, int32_t w = 10)
{
	return x + w*y; 
}

inline void index_to_grid(int32_t index, int32_t& x, int32_t& y, int32_t w = 10)
{
	x = index % w;
	y = index / w;
}

void create_grid_graph()
{
	SGVector<int32_t> card(2);
	card[0] = 2;
	card[1] = 2;
	SGVector<float64_t> w(4);
	w[0] = 0.0; // 0,0
	w[1] = 0.5; // 1,0
	w[2] = 0.5; // 0,1
	w[3] = 0.0; // 1,1
	SGString<char> tid(const_cast<char*>("pairwise"), 8);
	CTableFactorType* factortype = new CTableFactorType(tid, card, w);
	SG_REF(factortype);

	SGVector<int32_t> vc(100);
	SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, 2);
	CFactorGraph fg(vc);

	// Add factors
	for (int32_t x = 0; x < 10; x++)
	{
		for (int32_t y = 0; y < 10; y++)
		{
			if (x > 0)
			{
				SGVector<float64_t> data;
				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x,y,10);
				var_index[1] = grid_to_index(x-1,y,10);
				CFactor* fac1 = new CFactor(factortype, var_index, data);
				fg.add_factor(fac1);
			}

			if (y > 0)
			{
				SGVector<float64_t> data;
				SGVector<int32_t> var_index(2);
				var_index[0] = grid_to_index(x,y-1,10);
				var_index[1] = grid_to_index(x,y,10);
				CFactor* fac1 = new CFactor(factortype, var_index, data);
				fg.add_factor(fac1);
			}
		}
	}

	SG_UNREF(factortype);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	//sg_io->set_loglevel(MSG_DEBUG);
	
	create_grid_graph();

	exit_shogun();

	return 0;
}

