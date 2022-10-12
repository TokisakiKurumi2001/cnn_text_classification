from datasets import load_dataset, concatenate_datasets
dataset = load_dataset('csv', data_files=["data/train_wseg.csv"])
dataset['test'] = load_dataset('csv', data_files=["data/predict_test.csv"]).pop('train')
dataset['train'] = dataset['train'].rename_column("Categories", "label")
dataset = dataset.remove_columns('Unnamed: 0')
dataset = concatenate_datasets([dataset['train'], dataset['test']])