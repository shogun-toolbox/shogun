This cites the relevant points from
http://www.linux.org/docs/ldp/howto/Program-Library-HOWTO/shared-libraries.html
according to which we introduce ABI changes and thus have to change the soname
of libshogun/libshogunui.

1. The behavior of a function changes so that it no longer meets its original
specification,

2. Exported data items change (exception: adding optional items to the ends of
structures is okay, as long as those structures are only allocated within the
library).

3. An exported function is removed.

4. The interface of an exported function changes.

5. Add reimplementations of virtual functions (unless it it safe for older
binaries to call the original implementation), because the compiler evaluates
SuperClass::virtualFunction() calls at compile-time (not link-time).

6. Add or remove virtual member functions, because this would change the size
and layout of the vtbl of every subclass.

7. Change the type of any data members or move any data members that can be
accessed via inline member functions.

8. Change the class hierarchy, except to add new leaves.

9. Add or remove private data members, because this would change the size and
layout of every subclass.

10. Remove public or protected member functions unless they are inline.

11. Make a public or protected member function inline.

12. Change what an inline function does, unless the old version continues
working.

13. Change the access rights (i.e. public, protected or private) of a member
function in a portable program, because some compilers mangle the access rights
into the function name.
