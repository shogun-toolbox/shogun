#include <pthread.h>

int main()
{
	volatile pthread_spinlock_t spinlock;
	pthread_spin_init(&spinlock, 0);
	return 0;
}
