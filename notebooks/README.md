# Notebooks 使用说明

- 使用内核与依赖
  - 安装依赖与注册内核：`make install && make kernel`
  - 启动 JupyterLab：`make lab`（或 Notebook：`make nb`）
- 自动加入项目路径
  - 在笔记本首个代码单元加入：
```python
from dalong_catboost.notebook_utils import ensure_repo_in_sys_path
ensure_repo_in_sys_path()
```
- 清理输出（减少 Git 噪音）
  - 初始化：`make nbstrip-install`
  - 手动清理：`make nbstrip`
- 数据与产物
  - 原始/中间数据放在 `data/`
  - 训练模型放在 `models/`，其它产物在 `artifacts/`
