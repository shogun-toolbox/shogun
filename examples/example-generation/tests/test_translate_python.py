from translate import Translator
import json
import unittest

"""
Example assertions
self.assertEqual("asd", "asd")
self.assertRaises(TypeError, random.shuffle, (1,2,3))
self.assertTrue("key" in {"key": 2})
"""

class TestPythonTranslator(unittest.TestCase):

    def setUp(self):
        with open("targets/python.json", "r") as targetFile:
            self.translator = Translator(json.load(targetFile))

    def test_translateProgram(self):
        """
        CSVFile trainf("train.dat")
        RealFeatures feats_train(trainf)
        CSVFile testf("test.dat")

        Translates to:
        trainf = CSVFile("train.dat")
        feats_train = RealFeatures(trainf)
        testf = CSVFile("test.dat")
        """
        programAST = [
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "trainf"},{"ArgumentList": {"Expr": {"StringLiteral": "train.dat"}}}]}},
            {"Statement": {"Init": [{"ObjectType": "RealFeatures"}, {"Identifier": "feats_train"}, {"ArgumentList": {"Expr": {"Identifier": "trainf"}}}]}},
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "testf"}, {"ArgumentList": {"Expr": {"StringLiteral": "test.dat"}}}]}}
        ]

        translation = self.translator.translateProgram(programAST)

        self.assertEqual(translation, u"from modshogun import CSVFile, RealFeatures\n\ntrainf = CSVFile(\"train.dat\")\nfeats_train = RealFeatures(trainf)\ntestf = CSVFile(\"test.dat\")\n")

    def test_translateProgramWithNewlines(self):
        programAST = [
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "trainf"},{"ArgumentList": {"Expr": {"StringLiteral": "train.dat"}}}]}},
            {"Statement": "\n"},
            {"Statement": {"Init": [{"ObjectType": "RealFeatures"}, {"Identifier": "feats_train"}, {"ArgumentList": {"Expr": {"Identifier": "trainf"}}}]}},
            {"Statement": "\n"},
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "testf"}, {"ArgumentList": {"Expr": {"StringLiteral": "test.dat"}}}]}}
        ]

        translation = self.translator.translateProgram(programAST)

        self.assertEqual(translation, u"from modshogun import CSVFile, RealFeatures\n\ntrainf = CSVFile(\"train.dat\")\n\nfeats_train = RealFeatures(trainf)\n\ntestf = CSVFile(\"test.dat\")\n")

    def test_dependenciesString(self):
        programAST = [
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "trainf"},{"ArgumentList": {"Expr": {"StringLiteral": "train.dat"}}}]}},
            {"Statement": {"Init": [{"ObjectType": "RealFeatures"}, {"Identifier": "feats_train"}, {"ArgumentList": {"Expr": {"Identifier": "trainf"}}}]}},
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "testf"}, {"ArgumentList": {"Expr": {"StringLiteral": "test.dat"}}}]}}
        ]

        translation = self.translator.translateProgram(programAST)
        dependenciesString = self.translator.dependenciesString()
        self.assertEqual(dependenciesString, u"from modshogun import CSVFile, RealFeatures\n\n")

    def test_translateInitCopy(self):
        initAST = [
            {"ObjectType": "IntMatrix"},
            {"Identifier": "multiple_k"},
            {"Expr": {"MethodCall": [
                {"Identifier": "knn"},
                {"Identifier": "classify_for_multiple_k"}
            ]}}
        ]
        translation = self.translator.translateInit(initAST)
        self.assertEqual(translation, u"multiple_k = knn.classify_for_multiple_k()")

    def test_translateInitConstruct(self):
        initAST = [
            {"ObjectType": "MulticlassLabels"},
            {"Identifier": "labels"},
            {"ArgumentList": {
              "Expr": {"Identifier": "train_labels"}
            }}
        ]
        translation = self.translator.translateInit(initAST)
        self.assertEqual(translation, u"labels = MulticlassLabels(train_labels)")

    def test_translateInitConstructMultiple(self):
        initAST = [
            {"ObjectType": "EuclideanDistance"},
            {"Identifier": "distance"},
            {"ArgumentList": [
              {"Expr": {"Identifier": "feats_train"}},
              {"Expr": {"Identifier": "feats_test"}}
            ]}
        ]
        translation = self.translator.translateInit(initAST)
        self.assertEqual(translation, u"distance = EuclideanDistance(feats_train, feats_test)")

    def test_translateStatementAssign(self):
        stmtAST = {
            "Assign": [
                {"Identifier": "knn_train"},
                {"Expr":
                    {"BoolLiteral": "False"}
                }
            ]
        }
        translation = self.translator.translateStatement(stmtAST)
        self.assertEqual(translation, u"knn_train = False\n")

    def test_translateStatementExpr(self):
        stmtAST = {
            "Expr": {
              "MethodCall": [
                {"Identifier": "knn"},
                {"Identifier": "train"}
              ]
            }
        }

        translation = self.translator.translateStatement(stmtAST)
        self.assertEqual(translation, u"knn.train()\n")

    def test_translateStatementNewLine(self):
        stmtAST = "\n"
        translation = self.translator.translateStatement(stmtAST)
        self.assertEqual(translation, u"\n")

    def test_translateStatementPrint(self):
        stmtAST = {
            "Print": {"Expr": {"Identifier": "multiple_k"}}
        }

        translation = self.translator.translateStatement(stmtAST)

        self.assertEqual(translation, u"print multiple_k\n")

    def test_translateType(self):
        typeAST = {
            "ObjectType": "IntMatrix"
          }
        translation = self.translator.translateType(typeAST)

        self.assertEqual(translation, u"IntMatrix")

    def test_translateExprEnum(self):
        enumAST = {
            "Enum": [{"Identifier":"LIBLINEAR_SOLVER_TYPE"}, {"Identifier": "L2R_L2LOSS_SVC_DUAL"}]
        }
        translation = self.translator.translateExpr(enumAST)

        self.assertEqual(translation, u"L2R_L2LOSS_SVC_DUAL")
        self.assertTrue((u"LIBLINEAR_SOLVER_TYPE", u"L2R_L2LOSS_SVC_DUAL") in self.translator.dependencies["Enums"])

    def test_translateArgumentListEmpty(self):
        argumentListAST = []
        translation = self.translator.translateArgumentList(argumentListAST)

        self.assertEqual(translation, u"")

    def test_translateArgumentListSingleExpr(self):
        argumentListAST = {
            "Expr": {
              "MethodCall": [
                {"Identifier": "knn"},
                {"Identifier": "train"}
              ]
            }
        }
        translation = self.translator.translateArgumentList(argumentListAST)

        self.assertEqual(translation, u"knn.train()")

    def test_translateArgumentListMultipleExpr(self):
        argumentListAST =  [
            {"Expr": {"Identifier": "kernel"}}, 
            {"Expr": {"Identifier": "feats_train"}}, 
            {"Expr": {"Identifier": "zmean"}}, 
            {"Expr": {"Identifier": "labels"}}, 
            {"Expr": {"Identifier": "lik"}}
        ]
        translation = self.translator.translateArgumentList(argumentListAST)

        self.assertEqual(translation, u"kernel, feats_train, zmean, labels, lik")

    def test_translateProgramComment(self):
        programAST = [
            {"Comment": " This is a comment"}
        ]
        translation = self.translator.translateProgram(programAST)

        self.assertEqual(translation, u"# This is a comment\n")


if __name__ == '__main__':
    unittest.main()