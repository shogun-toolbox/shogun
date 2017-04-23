# Developing Shogun

This is a very basic list of things how to get started hacking Shogun.
Your first steps should be to

1. Compile from source, see [INSTALL.md](INSTALL.md).
2. Run the API examples, see [INTERFACES.md](INTERFACES.md), or create your own, see [EXAMPLES.md](EXAMPLES.md)
4. Run the [tests](#testing).

As we would like to avoid spending a lot of our time on explaining the same basic things many times, please **excessively** use the internet for any questions on the commands and tools needed.
If you feel that this readme is missing something, please send a patch! :)

# Quicklinks
 * [Shogun git development cycle](#devcycle)
 * [Testing](#testing)
 * [Buildfarm](#buildfarm)
 * [API documentation](#api)
 * [CMake options](#cmake)

# Shogun git development cycle <a name="devcycle"></a>
We use the [git flow](https://guides.github.com/introduction/flow/) workflow.
The steps are

0. Read the [guide](https://guides.github.com/).
1. Register on [GitHub](https://github.com/).
2. Fork the shogun repository.
3. Clone your fork, add the original shogun develop repository as a remote, and check out locally

        git clone https://github.com/YOUR_USERNAME/shogun
        cd shogun
        git remote add upstream https://github.com/shogun-toolbox/shogun
        git branch develop
        git checkout develop
        git pull --rebase upstream develop

   The steps until here only need to be executed once, with the exception being the last command: rebasing against the development branch.
   You will need to rebase everytime when the develop branch is updated.

4. Create a feature branch (from develop)

        git branch feature/BRANCH_NAME

5. Your code here: Fix bug or add feature. If you add something, or fix something, mention it in the `NEWS` file.
6. **Make sure (!)** that locally, your code **compiles**, it is **[tested](#testing)**, it complies to the code style described on the wiki.

        make && make test

    If something does not work, try to find out whether your change caused it, and why.
    Read error messages and use the internet to find solutions.
    Compile errors are the easiest to fix!
    If all that does not help, ask us.

7. Commit locally, using neat and informative commit messages, grouping commits, potentially iterate over more changes to the code,

        git commit FILENAME(S) -m "Fix issue #1234"
        git commit FILENAME(S) -m "Add feature XYZ"

    The [amend option](https://help.github.com/articles/changing-a-commit-message/) is your friend if you are updating single commits (so that they appear as one)

        git commit --amend FILENAME(S)

    If you want to group say the last three commits as one, [squash](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) them, for example

        git reset --soft HEAD~3
        git commit -m 'Clear commit message'

8. [Rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) against shogun's develop branch.
    This might cause rebase errors, which you need to [solve](https://help.github.com/articles/resolving-merge-conflicts-after-a-git-rebase/)

        git pull --rebase upstream develop

9. Push your commits to your fork

        git push origin feature/BRANCH_NAME

    If you squashed or amended commits after you had pushed already, you might be required to force push via using the `git push -f` option **with care**.

10. Send a [pull request](https://help.github.com/articles/about-pull-requests/) (PR) via GitHub.
    As described above, you can always **update** a pull request using the the `git push -f` option. Please **do not** close and send new ones instead, always update.

11. Once the PR is merged, keep an eye on the [buildfarm](#buildfarm) to see whether your patch broke something.

## Requirements for merging your PR
 * Read some [tips](http://blog.ploeh.dk/2015/01/15/10-tips-for-better-pull-requests/) on how to write good pull requests.
    Make sure you don't waste your (and our) time by not respecting these basic rules.
 * All tests pass (your pull request causes [automatic checks](#buildfarm)).
    We will not look at the patch otherwise.
 * The PR is small in terms of lines changes.
 * The PR is clean and addresses **one** issue.
 * The number of commits is minimal (i.e. one), the message is neat and clear.
 * If C++ code: it is covered by [tests](#testing), it doesn't leak memory, its [API](#api) is documented, [code style](https://github.com/shogun-toolbox/shogun/wiki/Code-style).
 * If API example: it has a clear scope, it is minimal, it looks polished, it has a passing [test](#testing)
 * If docs: clear, correct English language, spell-checked
 * If notebook: cell output is removed, template is respected, plots have axis labels.
 * Formatting notebooks/docs: Please every sentence in a single line.

# Testing <a name="testing"></a>
There are three types of tests that can be executed locally, C++ unit tests, running the API examples, and integration testing the results of the API examples.
To activate them locally, enable the `-DENABLE_TESTING=ON` cmake switch before running cmake. Which tests are activated depends on your configuration.
Adding a test in most cases requires to re-run `cmake`.
All activated tests can be executed with

    make && make test

The first `make` is necessary as some tests need to be generated and/or compiled first.

Sometimes, it is useful to run a single test, which can be done via [ctest](https://cmake.org/Wiki/CMake/Testing_With_CTest), for example

    ctest -R unit-LibSVR
    ctest -R generated_cpp-binary_classifier-kernel_svm
    ctest -R integration_meta_cpp-binary_classifier-kernel_svm -V

If a test name (or even the `make test` target) does not exist, this means that your configuration did not include it.

If you are interested in details how the test is executed (command, variables, directory), add the `-V` option.
Further details can be extracted from the `CMakeLists.txt` configuration files in the tests folder.

## C++ Unit tests
These are based on the [googletest](https://github.com/google/googletest) framework and are located in `tests/unit/`.
You can compile them with

    make shogun-unit-test

You can execute single tests via `ctest`, or via directly executing the unit test binary and passing it a filter, which gives a more grained control over which sub-tests are executed

    ./bin/shogun-unit-test --gtest_filter=GaussianProcessRegression.apply_*

Note that wildcards are allowed. Running single sub-tests is sometimes useful (i.e. for bug hunting)

    ./bin/shogun-unit-test --gtest_filter=GaussianProcessRegression.apply_apply_regression

### Debugging and Memory leaks
**All your C++ code and unit tests must be checked to not leak memory!**
You want to use a memory checker such as [valgrind](http://valgrind.org/) (or a debugger such as [gdb](https://www.gnu.org/software/gdb/)).
If you do that, you might want to compile with debugging symbols and without compiler optimizations, by using `-DCMAKE_BUILD_TYPE=Debug`

Then

    valgrind ./shogun-unit-test --gtest_filter=GaussianProcessRegression.apply_apply_regression
    gdb ./shogun-unit-test --gtest_filter=GaussianProcessRegression.apply_apply_regression

The option `--leak-check=full` for valgrind might be useful.
In addition to manually running valgrind on your tests, you can use `ctest` to check multiple tests.
This requires to be enable in dashboard reports in via `-DBUILD_DASHBOARD_REPORTS=ON`.
For example

    ctest -D ExperimentalMemCheck -R unit-GaussianProcessRegression

#### Adding tests
We aim to write clear, minimal, yet exhaustive tests of basic building blocks in Shogun.
Whenever you send us C++ code, we will ask you for a unit test for it.
We do test numerical results as compared to reference implementations (e.g. in Python), as well as corner cases, consistency etc.
Read on [test driven development](https://en.wikipedia.org/wiki/Test-driven_development), and search the web for tips on unit tests, e.g. [googletest's tips](https://github.com/google/googletest/blob/master/googletest/docs/Primer.md).

Take inspiration from existing tests when writing new ones.
Please structure them well.

## API example tests
Make sure to read [INTERFACES.md](INTERFACES.md) and [EXAMPLES.md](EXAMPLES.md) to understand how API examples are generated, you will need the cmake switch `-DBUILD_META_EXAMPLES=ON`.
Every API example is used for two tests: simple execution and continuous integration testing of results.
These two tests are executed for every enabled interface language.

Note that code for all interface examples needs to be generated as part of `make`, or using

    make meta_examples

This needs to be done everytime you add or modify an example.
Examples for compiled interface languages (e.g. C++, Java) need to be compiled, either as part of `make`, or via more specific targets, e.g.

    make build_cpp_meta_examples
    make build_java_meta_examples

Check the `CMakeLists.txt` in `examples/meta/*` for all such make targets.

### Simple execution.
These tests are to make sure the code is executable, and to generate results for integration testing.
These can be executed with `ctest` as described above, e.g.

    ctest -R generated*
    ctest -R generated_cpp-binary_classifier-kernel_svm -V

You can also execute the examples manually as described in [INTERFACES.md](INTERFACES.md).
Note that the `data` git submodule is required to run the examples, see [INSTALL.md](INSTALL.md).

Check the `CMakeLists.txt` in `examples/meta/*` for further details.

#### Adding tests
As every example is turned into a test when running `cmake`, all you need to do is to add an example as described in [EXAMPLES.md](EXAMPLES.md).

### Integration testing of results
You will note that each example produces an output file with the `*.dat` extension.
This is a serialized version of all numerical results of the example.
The purpose is to make sure all interface versions (say C++ and Python) of an example produce the same output, and that this output does not change over time.

The reference results are stored in the `data` git submodule, more precisely in `data/testsuite/meta/*`.
There is a symbolic link for both generated and reference results in the `build/tests/meta/` folder.
Naturally, these tests depend on executing the corresponding example first. Therefore, running a test does not run the example again, but it simply compares the output to the reference file.

Again `ctest` can be used,

    ctest -R integration_meta_*
    ctest -R integration_meta_cpp-binary_classifier-kernel_svm
    ctest -R integration_meta_python-binary_classifier-kernel_svm

See the `CMakeLists.txt` in `tests/meta` for details on the mechanics.

#### Adding tests
CMake automatically creates a test for every reference result file that it finds.
Therefore, if you want to add new test, for example after having added an example as described in [EXAMPLES.md](EXAMPLES.md), then you need to copy its generated output to the reference file folder, e.g.

    cp build/tests/meta/generated_results/cpp/regression/kernel_ridge_regression.dat data/testsuite/meta/regression/

Note we usually use the output of the C++ example as reference.

Once that is done, it would be good if you sent us a patch with the new test.
This is done via first sending a PR against the [shogun-data](https://github.com/shogun-toolbox/shogun-data), just like the standard [development cycle](#devcycle), after doing (in the `data` directory)

    git commit testsuite/meta/regression/kernel_ridge_regression.dat -m "Integration testing data for kernel ridge regression"
    git push origin

After this PR is merged, you need to send a second PR against the main repository, after commiting the updated version hash of the submodule (in the main shogun directory)

    git commit data -m "Updated to including kernel ridge regression test data"
    git push origin

If everything worked, then the [travis](#buildfarm) build in the second PR will include your test in all interface languages.
Please check the logs!

# Build farm <a name="devcycle"></a>
We run two types of buildfarms that are automatically triggered

1. [Travis](https://travis-ci.org/shogun-toolbox/shogun), and [AppVeyor](https://ci.appveyor.com/project/vigsterkr/shogun) executed in a third-party cloud when **opening** a PR
2. [Buildbot](http://buildbot.shogun-toolbox.org/waterfall), executed in our own cloud **after** every merged PR or commit

In addition, we have a few hooks on PRs that are executed along with travis, such as a preview of API examples.
You will see a list of checks in your PR.

## Travis
This is to do basic sanity checks on every PR. All interfaces have a different build, see `.travis.yml` in the repository.
The Docker image that runs the travis tests is based on `configs/shogun/Dockerfile` and can be found [here](https://hub.docker.com/r/shogun/shogun-dev/).

If you obbey the [dev cycle](#devcycle), in particular if you run tests before sending a PR, travis should never fail.

**If** travis fails

1. **Read the logs**, find the error message
2. Try to identify the problem
3. Find out whether you caused it
4. If so, reproduce locally
5. Fix it and update your PR

## Buildbot
This service builds and tests Shogun in a large number of different configurations, OS, interfaces, etc.
It ensures Shogun is portable, the build is backward compatible. It analysis Shogun's memory usage and performs static code analysis.
It often catches very subtle errors

After one of your PR is merged, check the status of the buildbot for a while.
The [waterfall](http://buildbot.shogun-toolbox.org/waterfall) view is most useful.
Again, check the logs if there are problems.


# CMake options for developers <a name="cmake"></a>
See also [INSTALL.md](INSTALL.md).
Options for developers (debugging symbols on, optimization off, etc.):

    cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON -DBUILD_DASHBOARD_REPORTS=ON ..

Options for building the final binaries (debugging off, optimization on):

    cmake -DCMAKE_BUILD_TYPE=Release ..

# API documentation <a name="api"></a>
Shogun uses [doxygen](http://www.stack.nl/~dimitri/doxygen/) for its [API documentation](shogun.ml/api).
Every bit of C++ code that is added to Shogun needs doxygen compatible source-code comments.

 * Every class needs a description of what it implements. If possible, use LaTeX for math.
 * Every method needs a description, plus all parameters and return values documented.

Check existing code for inspiration. Documentation is important, so polish as good as you can!

If you have doxygen installed, you can generate the documentation locally via running

    make doxygen

and then opening `build/doc/doxygen/html/index.html` with the browser.
