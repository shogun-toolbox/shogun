#ifndef _SHOGUN_EXCEPTION_H_
#define _SHOGUN_EXCEPTION_H_

/** class ShogunException */
class ShogunException {
	public:
		/** constructor
		 *
		 * @param str exception string
		 */
		ShogunException(const char* str);

		/** get exception string
		 *
		 * @return the exception string
		 */
		inline const char* get_exception_string() { return val; }

	private:
		/** exception string */
		char* val;
};

#endif // _SHOGUN_EXCEPTION_H_
