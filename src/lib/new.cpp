#include "lib/ShogunException.h"
#include "lib/new.h"
#include <stdio.h>

void* operator new(size_t size)
{
	void *p=malloc(size);
	if (!p)
		throw ShogunException("Assertion failed for new.\n");

	printf("Overloaded new, created object of size %d at %p.\n", size, p);
	return p;
}

void operator delete(void *p)
{
	printf("Free in overloaded delete at %p.\n", p);
	free(p);
}

void* operator new[](size_t size)
{
	void *p=malloc(size);
	if (!p)
		throw ShogunException("Assertion failed for new.\n");

	printf("Overloaded new[], created object of size %d at %p.\n", size, p);
	return p;
}

void operator delete[](void *p)
{
	printf("Free in overloaded delete[] at %p.\n", p);
	free(p);
}
