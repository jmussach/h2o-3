from __future__ import print_function
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator


def glrm_iris():
  print("Importing iris.csv data...")
  irisH2O = h2o.upload_file(pyunit_utils.locate("smalldata/iris/iris.csv"))
  irisH2O.describe()

  glrm_h2o = H2OGeneralizedLowRankEstimator(k=5, loss="Quadratic",transform="STANDARDIZE")
  glrm_h2o.train(x=irisH2O.names, training_frame=irisH2O)

  print("WoW")



if __name__ == "__main__":
  pyunit_utils.standalone_test(glrm_iris)
else:
  glrm_iris()
