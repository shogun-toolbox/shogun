from parse import FastParser, SlowParser
import unittest

class TestParse(unittest.TestCase):
    maxDiff = None
    def setUp(self):
        self.slowParser = SlowParser()
        self.fastParser = FastParser()

    def test_parseSlowEqualsFast1(self):
        program = """
        CSVFile trainf("train.dat")
        RealFeatures feats_train(trainf)
        CSVFile testf("test.dat")
        """

        slowParsedProgram = self.slowParser.parse(program)
        fastParsedProgram = self.fastParser.parse(program)

        self.assertEqual(slowParsedProgram, fastParsedProgram)

    def test_parseSlowEqualsFast2(self):
        program = """
        # Importing generated data
        CSVFile X("../data/gaussian_process_X.csv")
        CSVFile X_test("../data/gaussian_process_X_test.csv")
        CSVFile Y("../data/gaussian_process_Y.csv")
        CSVFile Y_test("../data/gaussian_process_Y_test.csv")

        RegressionLabels labels(Y)
        RegressionLabels labels_test(Y_test)
        RealFeatures feats_train(X)
        RealFeatures feats_test(X_test)

        # GP specification
        GaussianKernel kernel(10, 2)
        ZeroMean zmean()
        GaussianLikelihood lik()
        lik.set_sigma(0.5)
        ExactInferenceMethod inf(kernel, feats_train, zmean, labels, lik)

        # train GP
        GaussianProcessRegression gp(inf)
        gp.train()

        # some things we can do
        RealVector alpha = inf.get_alpha()
        RealVector diagonal = inf.get_diagonal_vector()
        RealMatrix cholesky = inf.get_cholesky()

        # get mean and variance vectors
        RealVector mean = gp.get_mean_vector(feats_test)
        RealVector variance = gp.get_variance_vector(feats_test)

        # plotting in python
        #from numpy import sqrt
        #from pylab import plot, show, legend, fill_between
        #plot(feats_train[0],labels,'x') # training observations
        #plot(feats_test[0],labels_test,'-') # ground truth of test
        #plot(feats_test[0],mean, '-') # mean predictions of test
        #fill_between(feats_test[0],mean-1.96*sqrt(variance),mean+1.96*sqrt(variance),color='grey')  # 95% confidence interval
        #legend(["training", "ground truth", "mean predictions"])

        #show()

        print "Alpha:"
        print alpha

        print "Diagonal:"
        print diagonal

        print "Variance:"
        print variance

        print "Mean:"
        print mean

        print "Cholesky:"
        print cholesky
        """

        slowParsedProgram = self.slowParser.parse(program)
        fastParsedProgram = self.fastParser.parse(program)

        self.assertEqual(slowParsedProgram, fastParsedProgram)

    def test_parseSlowEqualsFast3(self):
        program = """
        CSVFile train_data("../data/fm_train_real.dat")
        CSVFile test_data("../data/fm_test_real.dat")
        CSVFile label_data("../data/label_train_twoclass.dat")

        RealFeatures feats_train(train_data)
        RealFeatures feats_test(test_data)
        BinaryLabels labels(label_data)

        LibLinear svm(0.9, feats_train, labels)
        svm.set_liblinear_solver_type(enum LIBLINEAR_SOLVER_TYPE.L2R_L2LOSS_SVC_DUAL)
        svm.set_epsilon(0.001)
        svm.set_bias_enabled(True)
        svm.train()
        BinaryLabels predictions = svm.apply_binary(feats_test)

        print predictions.get_labels()
        """

        slowParsedProgram = self.slowParser.parse(program)
        fastParsedProgram = self.fastParser.parse(program)

        self.assertEqual(slowParsedProgram, fastParsedProgram)

if __name__ == '__main__':
    unittest.main()