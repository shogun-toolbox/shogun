GETTING STARTED
---------------

Shogun is split up into libshogun which contains all the machine learning
algorithms and 'static interfaces' helpers,
the static interfaces python_static, octave_static, matlab_static, r_static and
the modular interfaces python_modular, octave_modular and r_modular (all found
in the src/interfaces/ subdirectory with corresponding name). See src/INSTALL
on how to install shogun.

In case one wants to extend shogun the best way is to start using its library.
This can be easily done as a number of examples in examples/libshogun document.

The simplest libshogun based program would be

```
#include <base/init.h>

int main(int argc, char** argv)
{
    init_shogun();
    exit_shogun();
    return 0;
}
```

which could be compiled with `g++ -I/usr/include/shogun -lshogun minimal.cpp
-o minimal` and obviously does nothing (apart form initializing and destroying
a couple of global shogun objects internally).

In case one wants to redirect shoguns output functions SG_DEBUG, SG_INFO,
SG_WARN, SG_ERROR, SG_PRINT etc, one has to pass them to init_shogun() as
parameters like this

```
void print_message(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

void print_warning(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

void print_error(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

init_shogun(&print_message, &print_warning,
					&print_error);
```

To finally see some action one has to include the appropriate header files,
e.g. we create some features and a gaussian kernel


```
#include <labels/Labels.h>
#include <features/DenseFeatures.h>
#include <kernel/GaussianKernel.h>
#include <classifier/svm/LibSVM.h>
#include <base/init.h>
#include <lib/common.h>
#include <io/SGIO.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun(&print_message);

	// create some data
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++)
		matrix.matrix[i]=i;

	// create three 2-dimensional vectors 
	// shogun will now own the matrix created
	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>();
	features->set_feature_matrix(matrix);

	// create three labels
	CBinaryLabels* labels=new CBinaryLabels(3);
	labels->set_label(0, -1);
	labels->set_label(1, +1);
	labels->set_label(2, -1);

	// create gaussian kernel with cache 10MB, width 0.5
	CGaussianKernel* kernel = new CGaussianKernel(10, 0.5);
	kernel->init(features, features);

	// create libsvm with C=10 and train
	CLibSVM* svm = new CLibSVM(10, kernel, labels);
	svm->train();

	// classify on training examples
	for (int32_t i=0; i<3; i++)
		SG_SPRINT("output[%d]=%f\n", i, svm->apply_one(i));

	// free up memory
	SG_UNREF(svm);

	exit_shogun();
	return 0;

}
```

Now you probably wonder why this example does not leak memory. First of all,
supplying pointers to arrays allocated with new[] will make shogun objects own
these objects and will make them take care of cleaning them up on object
destruction. Then, when creating shogun objects they keep a reference counter
internally. Whenever a shogun object is returned or supplied as an argument to
some function its reference counter is increased, for example in the example
above

```
CLibSVM* svm = new CLibSVM(10, kernel, labels);
```

increases the reference count of kernel and labels. On destruction the
reference counter is decreased and the object is freed if the counter is <= 0.

It is therefore your duty to prevent objects from destruction if you keep a
handle to them globally *which you still intend to use later*. In the example
above accessing labels after the call to SG_UNREF(svm) will cause a
segmentation fault as the Label object was already destroyed in the SVM
destructor. You can do this by SG_REF(obj). To decrement the reference count of
an object, call SG_UNREF(obj) which will also automagically destroy it if the
counter is <= 0 and set obj=NULL only in this case.


Generally, all shogun C++ Objects are prefixed with C, e.g. CSVM and derived
from CSGObject. Since variables in the upper class hierarchy, need to be
initialized upon construction of the object, the constructor of base class
needs to be called in the constructor, e.g. CSVM calls CKernelMachine,
CKernelMachine calls CClassifier which finally calls CSGObject.

For example if you implement your own SVM called MySVM you would in the
constructor do

```
class MySVM : public CSVM
{
    MySVM( ) : CSVM()
    {
        ...
    }
};
```

In case you got your object working we will happily integrate it into shogun
provided you follow a number of basic coding conventions detailed below (see
FORMATTING for formatting instructions, MACROS on how to use and name macros,
TYPES on which types to use, FUNCTIONS on how functions should look like and
NAMING CONVENTIONS for the naming scheme.

CODING CONVENTIONS:
-------------------

*FORMATTING:*

- indenting uses stroustrup style with tabsize 4, i.e. for emacs use in your
	~/.emacs

  ```
	(add-hook 'c-mode-common-hook
	 (lambda ()
	  (show-paren-mode 1)
	  (setq indent-tabs-mode t)
	  (c-set-style "stroustrup")
	  (setq tab-width 4)))
  ```


	for vim in ~/.vimrc

  ```
	set cindent         " C style indenting
	set ts=4            " tabstop
	set sw=4            " shiftwidth
  ```

- for newlines use LF only; avoid CRLF and CR. Git can be configured to convert
  all newlines to LF as source files are commited to the repo by:

  ```
  git config --global core.autocrlf input
  ```

  (for more information consult http://help.github.com/line-endings/)

- avoid trailing whitespace (spaces & tabs) at end of lines and never use spaces
  for indentation; only ever use tabs for indentations.

  for emacs:

  ```
  (add-hook 'before-save-hook 'delete-trailing-whitespace)
  ```

  for vim in ~/.vimrc (implemented as an autocmd, use wisely):

  ```
  autocmd BufWritePre * :%s/\s\+$//e
  ```

- semicolons and commas ;, should be placed directly after a variable/statement

  ```
  x+=1;
  set_cache_size(0);

  for (uint32_t i=0; i<10; i++)
      ...
  ```

- brackets () and (greater/lower) equal sign ><= should should not contain
  unecessary spaces, e.g:

  int32_t a=1;
  int32_t b=kernel->compute();

  ```
  if (a==1)
  {
  }
  ```

  exceptions are logical subunits

  ```
  if ( (a==1) && (b==1) )
  {
  }
  ```

- avoid the use of inline functions where possible (little to zero performance
		impact). nowadays compilers automagically inline code when
		beneficial and within the same linking process

- breaking long lines and strings
	limit yourselves to 80 columns

	```
	for (int32_t vec=params->start; vec<params->end &&
			!CSignal::cancel_computations(); vec++)
	{
		//foo
	}
	```

	however exceptions are OK if readability is increased (as in function
	definitions)

- don't put multiple assignments on a single line

- functions look like

	```
	int32_t* fun(int32_t* foo)
	{
		return foo;
	}
	```

  and are separated by a newline, e.g:

	```
	int32_t* fun1(int32_t* foo1)
	{
		return foo;
	}
	```

	```
	int32_t* fun2(int32_t* foo2)
	{
		return foo2;
	}
	```

- same for if () else clauses, while/for loops

	```
	if (foo)
		do_stuff();

	if (foo)
	{
		do_stuff();
		do_more();
	}
	```

- one empty line between { } block, e.g.

	```
	for (int32_t i=0; i<17; i++)
	{
		// sth
	}

	x=1;
	```

*MACROS & IFDEFS:*

- use macros sparingly
- avoid defining constants using macros (bye bye typechecking), use

```
const int32_t FOO=5;
```

or enums (when defining several realted constants)

instead

- use ifdefs sparingly (really limit yourself to the ones necessary) as their
  extreme usage makes the code completely unreadable. to achieve that it may be
  necessary to wrap a function of (e.g. for
  pthread_create()/CreateThread()/thread_create() a wrapper function to create
  a thread and inside of it the ifdefs to do it the solaris/win32/posix way)
- if you need to use ifdefs always comment the corresponding #else / #endif
  in the following way:

```
#ifdef HAVE_LAPACK
  ...
#else //HAVE_LAPACK
  ...
#endif //HAVE_LAPACK
```

*TYPES:*
- types (use only these!):
	```
	char		(8bit char(maybe signed or unsigned))
	uint8_t		(8bit unsigned char)
	uint16_t	(16bit unsigned short)
	uint32_t	(32bit unsinged int)
	int32_t		(32bit int)
	int64_t		(64bit int)
	float32_t	(32bit float)
	float64_t	(64bit float)
	floatmax_t	(96bit or 128bit float depending on arch)
	```

	exceptions: file IO / matlab interface

- classes must be (directly or indirectly) derived from CSGObject

- don't use fprintf/printf, but SG_DEBUG/SG_INFO/SG_WARNING/SG_ERROR/SG_PRINT
  (if in a from CSGObject derived object) or the static SG_SDEBUG/... functions
  elsewise

*FUNCTIONS:*

- Functions should be short and sweet, and do just one thing.  They should fit
  on one or two screenfuls of text (the ISO/ANSI screen size is 80x24, as we
  all know), and do one thing and do that well.
- Another measure of the function is the number of local variables.  They
  shouldn't exceed 5-10, or you're doing something wrong.  Re-think the
  function, and split it into smaller pieces.  A human brain can
  generally easily keep track of about 7 different things, anything more
  and it gets confused.  You know you're brilliant, but maybe you'd like
  to understand what you did 2 weeks from now.

*GETTING / SETTING OBJECTS*

If a class stores a pointer to an object it should call SG_REF(obj) to increase
the objects reference count and SG_UNREF(obj) on class desctruction (which will
decrease the objects reference count and call the objects destructor if
ref_count()==0. Note that the caller (from within C++) of any get_* function
returning an object should also call SG_UNREF(obj) when done with the object.
This makes the swig wrapped interfaces automagically take care of object
destruction.

If a class function returns a new object this has to be stated in the
corresponding swig .i file for cleanup to work, e.g. if apply() returns a new
CLabels then the .i file should contain ```%newobject CClassifier::apply();```

*NAMING CONVENTIONS:*

- naming variables:
	- in classes are member variables are named like m_feature_vector (to
	  avoid shadowing and the often hard to find bugs shadowing causes)
	- parameters (in functions) shall be named e.g. feature_vector
	- don't use meaningless variable names, it is however fine to use
	  short names like i,j,k etc in loops
	- class names start with 'C', each syllable/subword starts with a
	  capital letter, e.g. CStringFeatures

- constants/defined objects are UPPERCASE, i.e. REALVALUED

- function are named like get_feature_vector() and should be limited to as few
  arguments as possible (no monster functions with > 5 arguments please)

- objects which can deal with features of type DREAL and class SIMPLE don't
  need to contain Real/Dense in class name

- others are required to clarify class/type they can handle, e.g.
  CSparseByteLinearKernel, CSparseGaussianKernel


- variable and function names are all lowercase (except for class
  Con/Destructors) syllables/subwords are separated by '_', e.g.
  compute_kernel_value(), my_local_variable

- class member variables all start with m_, e.g. m_member (this is to avoid
  shadowing)

- features and preprocessors are prefixed with featureclass (e.g. Dense/Sparse)
  followed by featuretype (Real/Byte/...)

*VERSIONING SCHEME:*

The git repo for the project is hosted on GitHub at
https://github.com/shogun-toolbox/shogun. To get started, create your own fork
and clone it ([howto](https://help.github.com/articles/fork-a-repo
"GitHub help - Fork a repo")).
Remember to set the upstream remote to the main repo by:

```
git remote add upstream git://github.com/shogun-toolbox/shogun.git
```

Its recommended to create local branches, which are linked to branches from
your remote repository.  This will make "push" and "pull" work as expected:

```
git checkout --track origin/master
git checkout --track origin/develop
```

Each time you want to develop new feature / fix a bug / etc consider creating
new branch using:

```
git checkout -b new_feature_name
```

While being on new_feature_name branch, develop your code, commit things and do
everything you want.

Once your feature is ready (please consider larger commits that keep shogun in
compileable state), rebase your new_feature_name branch on upstream/develop
with:

```
git fetch upstream
git checkout develop
git rebase upstream/develop
git checkout new_feature_name
git rebase develop
```

Now you can push it to your origin repository:

```
git push
```

And finally send a pull request (PR) to the develop branch of the shogun
repository in github.


- Why rebasing?

  What rebasing does is, in short, "Forward-port local commits to the updated
  upstream head". A longer and more detailed illustration with nice figures
  can be found at http://book.git-scm.com/4_rebasing.html. So rebasing (instead
  of merging) makes the main "commit-thread" of the repo a simple series.

  Rebasing before issuing a pull request also enable us to find and fix any
  potential conflicts early at the developer side (instead of at the one who
  merges your pull request).

- Multiple pull requests

  You can have multiple pull requests by creating multiple branches. Github
  only tracks the branch names you used for identify the pull request. So when
  you push new commits to your remote branch at github, the pull request will
  "update" accordingly.

- Non-fast-forward error

  This error happens when:

1. ```git checkout -b my-branch```
2. ... do something ...
3. ... rebasing ...
4. ```git push origin my-branch```
5. ... do more thing ...
6. ... rebasing ...
7. ```git push origin my-branch```

  then git will complain about non-fast-forward error and not pushing into the
  remote my-branch branch. This is because the first push has already created
  the my-branch branch in origin. Later when you run rebasing, which is a
  destructive operation for the local history. Since the local history is no
  longer the same as those in the remote branch, pushing is not allowed.

  Solution for this situation is to delete your remote branch by

```
    git push origin :my-branch
```

  and push again by

```
    git push origin my-branch
```

  note deleting your remote branch will not delete your pull request associated
  with that branch. And as long as you push your branch there again, your pull
  request will be OK.

- Unit testing/Pre-commit hook
  As shogun-toolbox is getting bigger and bigger code-reviews of pull requests
  are getting harder and harder. In order to avoid breaking the functionality
  of the existing code, we highly encourage contributors of shogun to use the
  supplied unit testing, that is based on Google C++ Mock Framework.

  In order to be able to use the unit testing framework one will need to have
  Google C++ Mock Framework installed on her machine as well as detected by
  the ./configure script. So, please check the produced configure.log whether
  the script detected it.

  Once it's detected if you add new classes to the code please define some basic
  unit tests for them under ./tests/unit (see some of the examples under that
  directory). As one can see the naming convention for files that contains the
  unit tests are: <classname>_unittest.cc

  Before commiting or sending a pull request please run 'make unit-tests' under
  ./src in order to check that nothing has been breaked by the modifications and
  the library is still acting as it's intended.

  One possible way to do this automatically is to add into your pre-commit hook
  the following code snippet (.git/hook/pre-commit):

```
#!/bin/sh

# run unit testing for basic checks
# and only let commiting if the unit testing runs successfully
cd src && make unit-tests
```

  This way before each commit the unit testing will run automatically and if it
  fails it won't let you commit until you don't fix the problem (or remove the
  pre-commit script :P

  Note that the script should be executable, i.e.
  ```
  chmod +x .git/hook/pre-commit
  ```
To make a release, adjust the [NEWS](NEWS) file properly, i.e. date, release
version (like 3.0.0), adjust the soname if required (cf. [README_soname]
(README_soname.md)) and if a new data version is required add that too. If
parameters have been seen changes increase the parameter version too.
