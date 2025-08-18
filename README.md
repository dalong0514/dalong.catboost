# dalong.catboost

一个最小可用的 CatBoost 项目骨架，支持配置化训练与命令行入口。

## 安装

```bash
pip install -e .[dev]
```

## 目录结构

```
dalong.catboost/
  ├─ src/dalong_catboost/
  │  ├─ __init__.py
  │  ├─ cli.py            # 命令行入口：训练
  │  ├─ config.py         # YAML 配置加载
  │  └─ model.py          # CatBoost 构建/训练/保存
  ├─ configs/
  │  └─ example_binary.yaml
  ├─ tests/
  │  └─ test_config.py
  ├─ pyproject.toml
  ├─ README.md
  ├─ LICENSE
  └─ .gitignore
```

## 快速开始

1. 准备训练数据 `data/train.csv`，包含目标列（例如 `label`）
2. 复制并修改 `configs/example_binary.yaml`
3. 运行训练

```bash
dalong-catboost --config configs/example_binary.yaml --out models/catboost.cbm
```

## 配置示例说明（节选）

```yaml
data:
  train_csv: data/train.csv
  target: label
  categorical: []
model:
  iterations: 200
  learning_rate: 0.1
fit:
  verbose: false
```

## 备注

- 需要 Python >= 3.9
- 推荐创建虚拟环境后再安装
