# 水排序高光剪辑流水线

本项目用于从水排序游戏原始视频中自动提取高光片段，并批量生成不同节奏的高光成品视频。

流水线分为三阶段：

1. 阶段一：粗筛，使用 CV 运动信号从原始视频中切出候选片段。
2. 阶段二：精筛，调用 OpenRouter 视觉模型判断片段是否为有效倒水或通关胜利。
3. 阶段三：组装，读取 `scored_segments.json` 并使用 FFmpeg concat 无损拼接生成成品。

## 环境准备

进入项目根目录：

```powershell
cd d:\app\project\AI_learn\Video\water_sort_highlights
```

推荐使用 Conda 创建环境：

```powershell
conda env create -f environment.yml
conda activate water-sort-highlights
```

如果不使用 Conda，至少需要安装 Python 依赖和 FFmpeg：

```powershell
pip install opencv-python numpy requests python-dotenv loguru tqdm openai tenacity
```

确认 FFmpeg 可用：

```powershell
ffmpeg -version
```

## 配置 API Key

复制环境变量示例文件：

```powershell
copy .env.example .env
```

然后编辑 `.env`，填写 OpenRouter 配置：

```text
OPENROUTER_API_KEY=你的 OpenRouter API Key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemini-3-flash-preview
```

## 目录约定

原始视频默认放在：

```text
data/raw/
```

示例：

```text
data/raw/level2/level2.mp4
data/raw/level3/level3.mp4
```

阶段一、二的中间产物会写入：

```text
data/interim/clips/{video_name}/segments.json
data/interim/clips/{video_name}/scored_segments.json
```

阶段三成品会写入：

```text
data/processed/
```

## 一键运行完整流程

处理 `data/raw` 下的直接视频文件：

```powershell
python src\main.py
```

递归处理 `data/raw` 下所有子目录中的视频：

```powershell
python src\main.py --recursive
```

只处理 `data/raw\level4` 子目录：

```powershell
python src\main.py --raw-subdir level1
```

指定粗筛分析帧率：

```powershell
python src\main.py --recursive --fps 5
```

指定成品输出目录：

```powershell
python src\main.py --recursive --processed-dir data\processed
```

固定阶段三盲盒随机种子，方便复现同一批结果：

```powershell
python src\main.py --recursive --stage3-seed 42
```

## 阶段一：粗筛

阶段一会从原始视频中切出候选片段，并生成 `segments.json`。

运行默认粗筛：

```powershell
python src\stage1_coarse.py
```

通过主入口运行阶段一到阶段二，但跳过阶段三：

```powershell
python src\main.py --skip-stage3
```

只运行阶段一粗筛，不调用阶段二模型，也不生成阶段三成品：

```powershell
python src\main.py --raw-subdir test1 --recursive --skip-stage2 --skip-stage3 --skip-stage4
```

处理子目录并递归扫描：

```powershell
python src\main.py --raw-dir data\raw --raw-subdir level2 --recursive --skip-stage3
```

## 阶段二：精筛

阶段二读取：

```text
data/interim/clips/{video_name}/segments.json
```

并输出：

```text
data/interim/clips/{video_name}/scored_segments.json
```

对所有已有 `segments.json` 执行阶段二：

```powershell
python src\stage2_fine.py
```

只处理 `level2`：

```powershell
python src\stage2_fine.py --video-name level2
```

指定模型：

```powershell
python src\stage2_fine.py --model google/gemini-3-flash-preview
```

跳过阶段一，只用现有 `segments.json` 执行阶段二：

```powershell
python src\main.py --only-stage2
```

只对已有 `level4` 的 `segments.json` 执行阶段二：

```powershell
python src\main.py --only-stage2 --raw-subdir test1
```

## 阶段三：极速组装

阶段三读取：

```text
data/interim/clips/{video_name}/scored_segments.json
```

并输出每个视频 10 个成品到：

```text
data/processed/
```

直接运行阶段三：

```powershell
python src\stage3_edit.py
```

只处理 `level2`：

```powershell
python src\stage3_edit.py --video-name level4
```

固定盲盒随机种子：

```powershell
python src\stage3_edit.py --seed 42
```

指定输出目录：

```powershell
python src\stage3_edit.py --processed-dir data\processed
```

通过主入口跳过阶段一、二，只执行阶段三：

```powershell
python src\main.py --only-stage3 --stage3-seed 42
```

只对已有 `level4` 的 `scored_segments.json` 执行阶段三：

```powershell
python src\main.py --only-stage3 --raw-subdir level4 --stage3-seed 42
```

## 阶段三输出命名

每个原始视频默认生成 10 个成品：

```text
{video_name}_01_sequential.mp4
{video_name}_02_comeback.mp4
{video_name}_03_panoramic.mp4
{video_name}_04_blindbox_01.mp4
...
{video_name}_10_blindbox_07.mp4
```

策略说明：

- `sequential`：顺产型，取最早的 6 个有效倒水片段，加胜利片段。
- `comeback`：逆袭型，取最晚的 6 个有效倒水片段，加胜利片段。
- `panoramic`：全景型，将有效倒水片段均分为 6 个区间，每区间取 1 个，加胜利片段。
- `blindbox_01` 到 `blindbox_07`：盲盒型，随机抽取并打乱 6 个有效倒水片段，加胜利片段。

如果存在通关胜利片段，阶段三会取时间最晚的胜利片段放在所有成品最后。

## 常用示例

已经完成阶段一、二，只想重新生成成品：

```powershell
python src\main.py --only-stage3 --stage3-seed 42
```

只重新生成 `level3` 的成品：

```powershell
python src\stage3_edit.py --video-name level3 --seed 42
```

重新用模型精筛 `level2`，再组装所有视频：

```powershell
python src\stage2_fine.py --video-name level2
python src\main.py --only-stage3 --stage3-seed 42
```

完整批量处理，并递归扫描所有原始视频：

```powershell
python src\main.py --recursive --stage3-seed 42
```

## 常见问题

如果提示 `ModuleNotFoundError: No module named 'loguru'`，说明当前 Python 环境缺少依赖：

```powershell
pip install loguru
```

如果提示找不到 FFmpeg，请先确认 FFmpeg 已安装并加入 PATH：

```powershell
ffmpeg -version
```

如果阶段三没有生成视频，请确认对应目录中存在：

```text
data/interim/clips/{video_name}/scored_segments.json
```

并且其中至少有 `selected: true` 的片段记录。


level 4运行
```powershell
python src/stage4_music.py --input-dir "D:\wjl\project\video\video\data\processed\batch_block_combinations\20260507_172344" --music-reuse-count 20 --workers 3
```