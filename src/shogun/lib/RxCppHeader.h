#ifndef SHOGUN_RXCPPHEADER_H
#define SHOGUN_RXCPPHEADER_H

/**
* Rx namespace
*/
namespace rxcpp
{
	template <class, class, class, class, class>
	class observer;

	namespace subjects
	{
		template <class>
		class subject;
	}

	template <class>
	class dynamic_observable;
	template <class, class>
	class observable;
	template <class, class>
	class subscriber;
	class subscription;
}

#endif // SHOGUN_RXCPPHEADER_H
