/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_NEIGHBOR_CALLBACK_H_
#define TAPKEE_NEIGHBOR_CALLBACK_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/tapkee_defines.hpp>
/* End of Tapkee includes */

template<class RandomAccessIterator>
struct neighbors_finder
{
	virtual Neighbors find_neighbors(RandomAccessIterator begin, RandomAccessIterator end, IndexType k);
};

#endif
