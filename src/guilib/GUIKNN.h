#ifndef _GUIKNN_H__
#define _GUIKNN_H__ 

#include "classifier/KNN.h"
#include "features/Labels.h"

class CGUI;

class CGUIKNN
{

public:
	CGUIKNN(CGUI* g);
	~CGUIKNN();

	bool new_knn(CHAR* param);
	bool train(CHAR* param);
	bool test(CHAR* param);
	bool load(CHAR* param);
	bool save(CHAR* param);

 protected:
	CGUI* gui;
	CKNN* knn;
	int k;
};
#endif
