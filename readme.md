# 2019 bytedance icme 短视频内容理解与推荐竞赛

### 代码结构
├── data_io.py  
├── features.py  
├── input  
│   ├── final_track2_test_no_anwser.txt  
│   ├── final_track2_train.txt  
│   ├── track2_face_attrs.txt  
│   └── track2_title.txt  
├── output  
│   └── result.csv  
├── readme.md  
└── train.py  

### 运行顺序
1. 创建文件夹input和output
2. 下载数据集并解压到文件夹input
3. 运行train.py训练模型
4. 生成的结果在output文件夹下面

### LB成绩
1. 2019-02-14: finish=0.65, like=0.88
2. 2019-02-15: finish=0.67, like=0.90

### Todo 
1. 使用深度学习对视频Title进行学习
2. 使用深度学习对视频特征进行学习