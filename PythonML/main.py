import module_ml

def main():

    dataframe = module_ml.read_dataset('iris-last.csv')
    selected_dataframe = module_ml.get_first_rows(dataframe)
    module_ml.save_plot_dataframe(selected_dataframe)
    encoded_dataframe = module_ml.encode_label(selected_dataframe)
    projected_dataframe = module_ml.select_columns(encoded_dataframe)
    train_test_dataframe = module_ml.split_test_train(projected_dataframe)
    perceptron = module_ml.train_perceptron(train_test_dataframe)
    module_ml.test_perceptron(train_test_dataframe, perceptron)

if __name__ == "__main__":
    main()
