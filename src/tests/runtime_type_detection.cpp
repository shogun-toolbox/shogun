#include <typeinfo>
#include <stdio.h>

template<class T> class foobar
{
public:
	foobar(T a)
	{
		//const typeinfo i=typeid(a);
		printf("%s\n",typeid(a).name());
	}

	~foobar() {};
};

class bla
{
public:
	bla() {};
	~bla() {};
};

class blabla
{
public:
	blabla() {};
	~blabla() {};
};

void main()
{
	char* x;
	blabla* r;
	float y;
	typedef double REAL;
	REAL z;
	class foobar<REAL> a(z);
}
