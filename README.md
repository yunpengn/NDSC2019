# National Data Science Challenge 2019

This repository presents our approach to the problem in the [National Data Science Challenge 2019](https://careers.shopee.sg/ndsc/), organized by [Shopee](https://shopee.com/). We are a team of 4 undergraduate students from the [National University of Singapore](http://www.nus.edu.sg/), comprising of

- [Niu Yunpeng](http://github.com/yunpengn)
- [Jiang Chunhui](https://github.com/Adoby7)
- [Jin Shuyuan](https://github.com/CoderStellaJ)
- [He Yingxu](https://github.com/YingxuH)

## Problem Description

There are hundreds of thousands new products added to Shopee every day. To make relevant products easily discoverable, one fundamental challenge is to accurately extract relevant information from a large volume of products. In NDSC 2019, this real-world challenge of building an automatic solution to extract product related information through machine learning techniques is presented.

For the beginner category, we would focus on **Product Category Classification**. Participants are required to determine the category of a product given its image and title. Performance will be evaluated based on the accuracy of the classification results. More information can be found at the Kaggle competition page at [here](https://www.kaggle.com/c/ndsc-beginner).

## Setup Instructions

- Make sure you have installed [Python](https://www.python.org) 3.7.2, [pipenv](https://github.com/pypa/pipenv) and the best IDE [PyCharm](https://www.jetbrains.com/pycharm/). Otherwise, use the command below (applicable to MacOS only, assuming you have [Homebrew](https://brew.sh) installed already):
```bash
brew install python3
brew install pipenv
brew cask install pycharm
```
- Clone this repository by `git clone git@github.com:yunpengn/NDSC2019.git`
- Install the required dependencies by `pipenv install`
- Setup this project in PyCharm:
    - Make the `data` folder as _"excluded"_.
- Download `category.json`, `train.csv` and `test.csv` and move them to the `data` folder.
- To activate this project's virtualenv, run `pipenv shell`.
- To run a command inside the virtualenv, use `pipenv run python3 XXXX.py`.

## Data Preprocessing

- Download data online and put train.csv and test.csv into data folder

- Create image training set folder and title training set folder in data

- Run single script/process_validation.py

This will seperate data in train.csv into train and validation sets. actual_train.csv and validate.csv are generated in data folder.

- Run image_preprocessing.ipynb in data folder

This will divide data in actual_train.csv into different categories and create subfolders in image training set folder and title training set folder

## Licence

[GNU General Public Licence 3.0](LICENSE)
