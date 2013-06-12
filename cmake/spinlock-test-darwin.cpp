#include <libkern/OSAtomic.h>

int main()
{
	volatile OSSpinlock lock;
	OSSpinlockTry(&lock);
	return 0;
}
