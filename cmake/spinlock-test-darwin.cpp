#include <libkern/OSAtomic.h>

int main()
{
	volatile OSSpinLock lock;
	OSSpinLockTry(&lock);
	return 0;
}
