# NYC HVFHV 网约车数据分析与建模项目

本项目基于 2024 年纽约市 TLC HVFHV（High Volume For-Hire Vehicles）出行数据，完成了两个方向的任务：

1. 出行费用与时长预测
2. 拥堵模式挖掘与交通状态预测

项目当前以 Jupyter Notebook 形式组织，核心文件是 [part1.ipynb](/d:/mlpj/part1.ipynb) 和 [part2.ipynb](/d:/mlpj/part2.ipynb)。

## 项目内容

### Part 1: 费用与时长预测

[part1.ipynb](/d:/mlpj/part1.ipynb) 主要围绕 NYC HVFHV 行程级数据进行特征工程与监督学习建模，目标包括：

- `y_cost_no_tips`：乘客开销，不含小费，由 `base_passenger_fare + tolls + bcf + sales_tax + congestion_surcharge + airport_fee` 构造
- `y_time`：行程时长，对应 `trip_time`

这一部分完成了以下工作：

- 对 2024 年 1 至 12 月的原始 parquet 数据做年度概览统计
- 计算目标变量分位数，便于后续截尾和稳健建模
- 构建包含时间、空间、路线、平台编码、天气、节假日等信息的月度特征表
- 按时间切分数据集：
  - 训练集：1 至 9 月
  - 验证集：10 月
  - 测试集：11 至 12 月
- 使用 `XGBoost` 和 `LightGBM` 对费用、时长两个目标分别建模
- 采用抽样训练、目标截尾、`log1p` 变换与 early stopping 提升训练稳定性和效率

### Part 2: 拥堵模式分析

[part2.ipynb](/d:/mlpj/part2.ipynb) 关注城市交通运行状态，主要流程包括：

- 逐月读取 HVFHV 数据，只保留必要列以控制内存占用
- 按 `日期-小时-上车区域(PULocationID)` 聚合交通指标
- 融合天气与节假日数据
- 构建拥堵分析特征，如平均速度、速度波动、流量、天气影响等
- 使用 `K-Means` 对交通状态进行聚类，提炼拥堵模式
- 结合聚类标签，进一步使用 `RandomForestClassifier` 做交通状态预测
- 通过散点图、箱线图、混淆矩阵等方式解释模型结果

## 数据说明

项目默认使用 2024 年 NYC TLC HVFHV 月度 parquet 数据，notebook 中采用的默认路径为：

```text
/data/tlc_hvfhv_2024
```

配套外部数据包括：

- 天气数据：`/data/tlc_hvfhv_2024_out/weather_hourly_2024.parquet`
- 纽约州节假日数据：`/data/tlc_hvfhv_2024_out/holidays_NY_2024.parquet`

`part1.ipynb` 中保留了生成天气和节假日数据的示例代码，默认是注释状态，可按需启用。

## 项目结构

```text
.
├─ part1.ipynb                      # 行程级特征工程 + 费用/时长预测
├─ part2.ipynb                      # 拥堵聚类 + 交通状态预测
└─ 报告.pdf      # 题目说明/作业文档
```

在 notebook 默认配置下，运行过程中还会生成以下目录或文件：

```text
/data/tlc_hvfhv_2024_out/
├─ overview_2024.parquet
├─ overview_2024.csv
├─ target_quantiles_2024.parquet
├─ target_quantiles_2024.csv
├─ weather_hourly_2024.parquet
├─ holidays_NY_2024.parquet
└─ processed_congestion_data.parquet

/data/tlc_hvfhv_2024_features_wxhol/
└─ features_2024-01.parquet ... features_2024-12.parquet

/data/tlc_hvfhv_2024_models/
├─ xgb_cost_log_wxhol.json
├─ xgb_time_log_wxhol.json
├─ lgb_cost_log_wxhol.txt
└─ lgb_time_log_wxhol.txt
```

## 环境依赖

从 notebook 中实际使用的库来看，建议 Python 3.10+，并安装以下依赖：

```bash
pip install jupyter pandas polars numpy pyarrow matplotlib seaborn scikit-learn xgboost lightgbm joblib holidays requests
```

如果只运行 `part2.ipynb`，最少需要：

```bash
pip install jupyter pandas numpy pyarrow matplotlib seaborn scikit-learn
```

## 运行方式

### 1. 准备数据

- 将 2024 年 HVFHV 月度 parquet 文件放到 `/data/tlc_hvfhv_2024`
- 准备天气与节假日 parquet 文件，或使用 `part1.ipynb` 中的示例代码自行生成

原始数据文件命名格式应为：

```text
fhvhv_tripdata_2024-01.parquet
fhvhv_tripdata_2024-02.parquet
...
fhvhv_tripdata_2024-12.parquet
```

### 2. 启动 Notebook

```bash
jupyter notebook
```

### 3. 推荐执行顺序

1. 先运行 [part1.ipynb](/d:/mlpj/part1.ipynb)
2. 再运行 [part2.ipynb](/d:/mlpj/part2.ipynb)

推荐原因：

- `part1.ipynb` 会生成天气、节假日、年度统计、特征表等中间结果
- `part2.ipynb` 依赖天气和节假日 parquet，且会读取处理后的聚合结果继续建模

## 方法特点

- 使用时间切分而不是随机切分，避免未来信息泄露
- 对超大规模出行数据采用抽样训练和分月处理，兼顾内存与训练效率
- 使用 `Polars LazyFrame` 进行流式读取和特征构建，适合 parquet 大文件
- 同时考虑时空特征、平台特征、天气和节假日因素
- 将无监督聚类和有监督分类结合，用于解释并预测交通拥堵状态

## 注意事项

- notebook 中的路径以 Linux/服务器环境为默认配置，若在本地 Windows 运行，需要先修改数据目录
- 项目目前以 notebook 为主，还没有整理成独立的 Python 包或命令行脚本
- 由于数据量较大，建议在内存较充足的环境中运行
- `part1.ipynb` 中部分下载/生成数据代码为注释示例，使用前需要手动取消注释并检查依赖

## 后续可改进方向

- 将 notebook 重构为 `src/` + `configs/` 的工程化结构
- 增加 `requirements.txt` 或 `environment.yml`
- 在 README 中补充实验结果表格与可视化截图
- 增加模型评估结果落盘和自动化复现实验脚本

## 说明

这是一个以课程作业/数据挖掘实验为导向的项目，适合展示以下能力：

- 大规模出行数据处理
- 时空特征工程
- 回归与聚类/分类建模
- 基于外部数据的多源融合分析
