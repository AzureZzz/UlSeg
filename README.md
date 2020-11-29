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

2.Run train_val.py.

3.Run test.py.


| StageⅠ | StageⅡ |    DsC     |    IoU     |   DsC β    |   IoU β    | filter n |
| :----: | :----: | :--------: | :--------: | :--------: | :--------: | :------: |
|   B6   |   B6   | **0.8695** | **0.7897** |   0.8854   | **0.8188** |          |
|   B6   |   B7   |   0.8661   |   0.7857   |   0.8843   |   0.8161   |          |
|   B7   |   B6   |   0.8676   |   0.7874   | **0.8908** |   0.8161   |          |
|   B7   |   B7   |   0.8629   |   0.7806   |   0.8805   |   0.8106   |          |
| B6_GAN |   B6   |   0.8588   |   0.7778   |   0.8839   |   0.8149   |          |
| B7_GAN |   B6   |   0.8613   |   0.7804   |   0.8860   |   0.8123   |          |
| B6_GAN |   B7   |   0.8571   |   0.7754   |   0.8847   |   0.8107   |          |
| B7_GAN |   B7   |   0.8606   |   0.7788   |   0.8855   |   0.8078   |          |