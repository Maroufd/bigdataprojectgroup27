import argparse
import preprocessing
import models
import statistics

def run_application(config, *, seed=69, verbose=False):
    """Runs a simulation using dynamic routing.
    :param config: namespace with the configuration for the run
    :param seed: seed to use for the random cars
    :param verbose: whether or not to show user output
    """
    file=config.file
    trainsize=config.trainsize
    testsize=config.testsize
    model=config.model

    df=preprocessing.readandcreate(file)

    df=preprocessing.drop_forbidden(df)

    df=preprocessing.date_preprocess(df)

    df=preprocessing.drop_not_needed(df)

    df=preprocessing.filter_null(df)

    df.show(10)


    df=preprocessing.coltoint(df)

    df=preprocessing.convert_to_vector(df)

    #df= preprocessing.get_correlation_matrix(df,"features")

    df= preprocessing.create_features_vector(df)

    train_set,test_set=preprocessing.split_set(df,trainsize,testsize)

    if model=="LinearRegression":
        lr_model,trainingSummary, test_result=models.LinearRegressionmodel(train_set,test_set)

    statistics.print_linear_regression_summary(lr_model)

    statistics.print_training_summary(trainingSummary)

    statistics.print_test_summary(test_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="2008.csv.bz2")
    parser.add_argument("--trainsize", type=float, default=0.7)
    parser.add_argument("--testsize", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="LinearRegression")
    config = parser.parse_args()
    run_application(config, verbose=True)
