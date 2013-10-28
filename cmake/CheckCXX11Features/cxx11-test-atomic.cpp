#include <atomic>

int main()
{
	volatile std::atomic<int> x;
	x.store(0);

	bool ret = (
		x.load() == 0
	);

	return ret ? 0 : 1;
}