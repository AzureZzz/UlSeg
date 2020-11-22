# UlSeg
***
Original project link：https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st

We trained our data set using the scheme of the above project.

The experimental results are as follows：

## Experimental Result

***

| StageⅠ | TTAⅠ | StageⅡ | TTA Ⅱ |     Dice      |      IoU      |    Dice β     |     IoU β     |
| :----: | :--: | :----: | :---: | :-----------: | :-----------: | :-----------: | :-----------: |
|   √    |      |        |       |   0.8324247   |   0.7458447   |   0.8717661   |   0.8008780   |
|   √    |      |   √    |       |   0.8437420   |   0.7632147   |   0.8816358   |   0.8146432   |
|   √    |  √   |   √    |       |   0.8571215   |   0.7785505   |   0.8880194   |   0.8158846   |
|   √    |      |   √    |   √   |   0.8486418   |   0.7693327   |   0.8862817   |   0.8160666   |
|   √    |  √   |   √    |   √   | **0.8600747** | **0.7826157** | **0.8909538** | **0.8200555** |

***
## How to use?
1.Preparing your dataset.

├── dataset

│   ├── your_dataset

│   │   └── preprocessed

│   │       ├── stage1

|   |       |   ├── image

|   |       |   └── mask

│   │       ├── stage2

|   |       |   ├── image

|   |       |   └── mask

│   │       └── train.csv


2.Run train_val.py.

3.Run test.py.
