# Waytous Point Cloud Curation Pipeline

这是一个可直接运行的增强点云质量清洗工程骨架，按照中科慧拓（Waytous）场景实现：从 3D box 与点云提取结构化特征，输出质量分数，并将样本分流到 `keep`、`reject`、`human_review` 三类 JSONL manifest。

当前的 `HeuristicPlaceholderModel` 是可替换的占位模型，不代表真实生产模型或真实业务阈值。它让整条数据链路现在即可运行、测试和演示；接入训练好的分类器时只需实现 `QualityModel` 协议。

## 架构

```text
JSONL samples
    -> schema validation
    -> geometry feature extraction
       (dimensions / point count / density / ground offset / max 3D IoU)
    -> pluggable quality model (placeholder)
    -> confidence router
       |-- keep.jsonl
       |-- reject.jsonl
       `-- human_review.jsonl
    -> summary.json + lineage.json
```

## 快速开始

```bash
conda env create -f environment.yml
conda activate waytous-curation

waytous-curate \
  --input examples/samples.jsonl \
  --output output/demo \
  --config configs/default.json \
  --dataset-version demo-v1
```

也可不安装入口脚本直接运行：

```bash
PYTHONPATH=src python -m waytous_curation \
  --input examples/samples.jsonl \
  --output output/demo
```

运行测试：

```bash
pytest
```

## 输入格式

每行一个 JSON 对象：

```json
{
  "sample_id": "frame-001-car-01",
  "class_name": "car",
  "bbox": {"center": [0.0, 0.0, 0.75], "size": [4.2, 1.8, 1.5], "yaw": 0.0},
  "ground_z": 0.0,
  "points": [[0.0, 0.0, 0.5], [1.0, 0.4, 0.8]],
  "neighbor_boxes": [
    {"center": [5.0, 0.0, 0.75], "size": [4.0, 1.8, 1.5], "yaw": 0.0}
  ],
  "metadata": {"scene_id": "scene-001", "augmentation": "object_insert"}
}
```

`points` 也可以替换为 `points_path`，支持：

- `.json`：`[[x, y, z], ...]`
- `.jsonl`：每行 `[x, y, z]` 或 `{"x": ..., "y": ..., "z": ...}`
- `.csv`：带或不带表头的 `x,y,z`

路径相对于输入 manifest 所在目录解析。框内点数采用带 yaw 的 3D box 判断；3D IoU 当前采用轴对齐包围盒近似，便于在无第三方依赖下运行，生产接入时可替换为精确旋转框 IoU。

## 输出与可追溯性

每条输出记录都保留原始样本，附加：

- `curation.features`：尺寸、体积、框内点数、点密度、底面与地面偏移、最大 3D IoU
- `curation.quality_score`：占位模型输出的 `[0, 1]` 分数
- `curation.decision`：三路分流结果
- `curation.reasons`：便于人工复核的解释
- `curation.model_version`：模型版本

`summary.json` 汇总数量与比例；`lineage.json` 记录输入文件哈希、配置哈希、运行参数、时间和 Python 版本。写文件采用临时文件后原子替换，避免中途中断留下半成品 manifest。

## 替换真实模型

实现以下接口即可：

```python
class MyModel:
    version = "my-model-v1"

    def predict_quality(self, features, sample):
        return 0.93
```

然后将实例传给 `CurationPipeline(config, model=MyModel())`。建议真实模型返回“样本质量合格”的校准概率，并基于独立验证集分别确定 `reject_max_score` 与 `keep_min_score`。

## 工程边界

- 本仓库不包含真实公司数据、权重、内部阈值或专有实现。
- 配置中的类别尺寸范围和评分权重仅用于演示。
- 真实落地应增加旋转框精确 IoU、类别分层阈值、异常类型指标、MLflow/对象存储适配，以及基于人工复核结果的主动学习闭环。

