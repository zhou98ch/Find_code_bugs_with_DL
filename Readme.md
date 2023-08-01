## My local environment
python 3.8, torch 1.11.0
## Installation
```pip install gensim ```

```pip install libcst```#LibCST requires Python 3.7+ 

```pip install nltk ```

```pip install numpy```

```pip3 install torch torchvision torchaudio ```
#or go to <https://pytorch.org/get-started/locally/> to find the corresponding command for your local environment

## Usage
### To train the model with input data and get output of the model-configuration file(pth)
```python Train.py --source "shared_resources/data/functions_list.json" --destination model.pth ```

### To predict on unlabeled data and store the result in a json file
```python Predict.py --model 'model.pth' --source "shared_resources/real_test_for_milestone3/real_consistent.json" --destination t.json ```
