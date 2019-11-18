# data preparation

## BU-3DFE

To format the BU-3DFE database, ensure the root directory has the following structure (it should do from the vendor):

```
/my/bu/root/
│
└───F0001
│   │   F0001_AN01WH_F2D.bmp
│   │   F0001_AN02WH_F2D.bmp
│   │   ... 
└───F0002
│   │   F0001_AN01WH_F2D.bmp
│   │   F0001_AN02WH_F2D.bmp
...
```

## MultiPIE

To format the MultiPIE database, ensure the root directory has the following structure (it should do from the vendor):

```
/my/multiPIE/root/
├── 2 (id)
│   ├── 0 (pose)
│   │   ├── Disgust (emotion)
│   │   │   ├── 1
│   │   │   │   └── 002_03_03_051_01.jpg
│   │   │   ├── 2
│   │   │   │   └── 002_03_03_051_03.jpg
│   │   │   │   
│   │   │   ├── ...
│   │   │   │   
│   │   │   └── 5
│   │   │       └── 002_03_03_051_13.jpg
│   │   ├── Neutral
│   │   │   ├── 1
│   │   │   │   ├── 002_01_01_051_01.jpg
│   │   │   │   ├── ...
│   │   │   ├── ...
│   ├── ...
...
```

## RaFD

To format the RaFD database, ensure the root directory has the following structure (it should do from the vendor):

```
/my/rafd/root/
├── Rafd000_01_Caucasian_female_angry_frontal.jpg 
├── Rafd000_01_Caucasian_female_angry_left.jpg
├── ...