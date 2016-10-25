* SHOGUN Release version 4.1.0 (libshogun 17.1, data 0.10, parameter 1)
* This is a new feature and cleanup release:
* Features:
	- Added GEMPLP for approximate inference to the structured output framework [Jiaolong Xu].
	- Effeciency improvements of the FITC framework for GP inference (FITC_Laplce, FITC, VarDTC) [Wu Lin].
	- Added optimisation of inducing variables in sparse GP inference [Wu Lin].
	- Added optimisation methods for GP inference (Newton, Cholesky, LBFGS, ...) [Wu Lin].
	- Added Automatic Relevance Determination (ARD) kernel functionality for variational GP inference [Wu Lin].
	- Updated Notebook for variational GP inference [Wu Lin].
	- New framework for stochastic optimisation (l1/2 loss, mirror descent, proximal gradients, adagrad, SVRG, RMSProp, adadelta, ...) [Wu Lin].
	- New Shogun meta-language for automatically generating code listings in all target languages [Esben Sörig].
	- Added periodic kernel [Esben Sörig].
	- Add gradient output functionality in Neural Nets [Sanuj Sharma].
* Bugfixes:
	- Fixes for java_modular build using OpenJDK [Björn Esser].
	- Catch uncaught exceptions in Neural Net code [Khaled Nasr].
	- Fix build of modular interfaces with SWIG 3.0.5 on MacOSX [Björn Esser].
	- Fix segfaults when calling delete[] twice on SGMatrix-instances [Björn Esser].
	- Fix for building with full-hardening-(CXX|LD)FLAGS [Björn Esser].
	- Patch SWIG to fix a problem with SWIG and Python >= 3.5 [Björn Esser].
	- Add modshogun.rb: make sure narray is loaded before modshogun.so [Björn Esser].
	- set working-dir properly when running R (#2654) [Björn Esser].
* Cleanup, efficiency updates, and API Changes:
	- Added GPU based dot-products to linalg [Rahul De].
	- Added scale methods to linalg [Rahul De].
	- Added element wise products to linalg [Rahul De].
	- Added element-wise unary operators in linalg [Rahul De].
	- Dropped parameter migration framework [Heiko Strathmann].
	- Disabled Python integration tests by default [Sergey Lisitsyn, Heiko Strathmann].
	
[Release on GitHub](https://github.com/shogun-toolbox/shogun/releases/tag/shogun_4.1.0)
