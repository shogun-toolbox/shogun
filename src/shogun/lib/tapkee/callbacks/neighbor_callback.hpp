#ifndef TAPKEE_NEIGHBOR_CALLBACK_H_
#define TAPKEE_NEIGHBOR_CALLBACK_H_

#include "../defines.hpp"

template<class RandomAccessIterator>
struct neighbors_finder
{
	virtual Neighbors find_neighbors(RandomAccessIterator begin, RandomAccessIterator end, unsigned int k);
};

#endif
