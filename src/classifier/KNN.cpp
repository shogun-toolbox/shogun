#include "classifier/KNN.h"

CKNN::CKNN(): k=3
{
}


CKNN::~CKNN()
{
}

REAL* CKNN::test()
{
	assert(CKernelMachine::get_labels());
	for (int i=0; i<CKernelMachine::get_labels()->get_num_labels(); i++)
	{
	}
}

bool CKNN::load(FILE* srcfile)
{
}

bool CKNN::save(FILE* srcfile)
{
}
