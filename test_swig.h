#ifndef TEST_SWIG_HPP
#define TEST_SWIG_HPP

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <string>

namespace test {
	class Base : public std::enable_shared_from_this<Base> {
		public:
			Base() {}
			virtual ~Base() {}

			bool equals(const Base* v) { return true; }
			template<typename T,
				class X = typename std::enable_if_t<std::is_base_of_v<Base, T>>,
				class Z = void>
			void set(const std::string& s, std::shared_ptr<T> v) {
				value = v;
			}

			template<typename T,
				typename V = typename std::enable_if<!std::is_base_of_v<Base, typename std::remove_pointer_t<T>>, T>::type>
			void set(const std::string& s, T v) {
				std::cout << v << "\n";
			}

			virtual std::string name() const = 0;
		private:
			std::shared_ptr<Base> value;
	};

	class Machine: public Base {
		public:
		Machine(): Base() {}
		~Machine() override {}
		virtual std::string name() const { return "C"; }
		std::string own() const { return "works"; }
	};
}
#endif