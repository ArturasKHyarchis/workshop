import pathlib
import pandas as pd

#path to your data file
data_path = pathlib.Path(r'c:\_dev\workshop\data\data.csv')

#path to save the split data
test_data_path = data_path.parent.joinpath('test_data.csv')
val_data_path = data_path.parent.joinpath('val_data.csv')
train_data_path = data_path.parent.joinpath('train_data.csv')

#load the data
data =

#split the data
train_data =
val_data =
test_data =


#save the split data
test_data.to_csv(test_data_path, index=False, encoding='utf-8', sep=',', quotechar='"')
val_data.to_csv(val_data_path, index=False, encoding='utf-8', sep=',', quotechar='"')
train_data.to_csv(train_data_path, index=False, encoding='utf-8', sep=',', quotechar='"')


### Check split sizes and distributions
print(f'Train size: {train_data.shape[0]}')
print(f'Validation size: {val_data.shape[0]}')
print(f'Test size: {test_data.shape[0]}')

test_pivot = test_data.pivot_table(index='Sentiment', aggfunc='size')
val_pivot = val_data.pivot_table(index='Sentiment', aggfunc='size')
train_pivot = train_data.pivot_table(index='Sentiment', aggfunc='size')

print('Train distribution:')
print(train_pivot)
print('Validation distribution:')
print(val_pivot)
print('Test distribution:')
print(test_pivot)