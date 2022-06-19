# Federated-Grade-Prediction
===========================
利用联邦学习进行高校学生成绩预测研究，并使用Streamlit搭建了简易的可视化平台


可点击网址进入：https://share.streamlit.io/cathynwpu/fl/main/main_fedrep.py


###########环境依赖
见requirments.txt,展示如下：
python==3.8
streamlit==1.10.0
torch==1.10.2
torchvision==0.11.3

###########部署步骤
1. 一键安装所需包
    pip install -r requirements.txt
    
    
2. 运行程序
    streamlit run main_fedrep.py


###########目录结构描述
├── Readme.md                   // help
├── data                        // 数据
├── models                     // 模型
│   ├── language_uutils.py
│   ├── dev.json                // 开发环境
│   ├── experiment.json         // 实验
│   ├── index.js                // 配置控制
│   ├── local.json              // 本地
│   ├── production.json         // 生产环境
│   └── test.json               // 测试环境
├── data
├── doc                         // 文档
├── environment
├── gulpfile.js
├── locales
├── logger-service.js           // 启动日志配置
├── node_modules
├── package.json
├── app-service.js              // 启动应用配置
├── static                      // web静态资源加载
│   └── initjson
│       └── config.js         // 提供给前端的配置
├── test
├── test-service.js
└── tools
