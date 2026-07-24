# 重教轨迹平滑回放 SOP

目标：把一份 RM75 重教轨迹文本先做平滑和重采样，再按可复现参数回放到机械臂。  
当前标准脚本：`script/experiments/smooth_reteach_trajectory.py`

这份 SOP 以 `data/SJQ_data/6_18.txt` 的成功复现为基准模板。  
已核对过的关键结果是：

- 输入：`data/SJQ_data/6_18.txt`
- 输出点数：`7058`
- 规划时长：`40.0 s`
- 自动步进周期：`0.005668 s`

## 1. 先确认输入格式

脚本要求输入文件每一行都是这种格式：

```json
{"point":[-33,-7518,69510,23913,18367,71107,-111899]}
```

注意：

- `point` 必须存在
- `point` 必须正好有 7 个数
- 默认 `--joint-scale 0.001`，表示原始值会被换算成角度

如果文件里混有别的行，例如：

```json
{"gripper":1}
```

这种行必须先去掉，否则脚本会报：

```text
ValueError: line N: missing 'point' field
```

## 2. 标准成功模板

`6_18.txt` 对应的成功复现命令是：

```bash
cd /home/chenlvping/1_DM_work/vio_eval

python3 script/experiments/smooth_reteach_trajectory.py \
  --input /home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/6_18.txt \
  --duration-sec 40 \
  --output-json /home/chenlvping/1_DM_work/vio_eval/data/smoothed_robot_record.json \
  --output-txt /home/chenlvping/1_DM_work/vio_eval/data/smoothed_robot_record.txt
```

推荐把这条命令当作之后所有新轨迹的起点，只替换：

- `--input`
- `--output-json`
- `--output-txt`
- 必要时再改 `--duration-sec`

## 3. 为什么输出路径要写绝对路径

这个脚本对默认输出名有特殊处理。

如果你写成：

```bash
--output-json data/smoothed_robot_record.json
```

它仍可能把文件写到 `script/data/` 下，而不是仓库根目录的 `data/` 下。

所以 SOP 里统一要求：

- `--input` 用绝对路径
- `--output-json` 用绝对路径
- `--output-txt` 用绝对路径

这样最稳。

## 4. 新轨迹的推荐流程

### 4.1 先做 dry-run

第一次处理某条新轨迹时，先不要让机械臂真正运动：

```bash
python3 script/experiments/smooth_reteach_trajectory.py \
  --input /abs/path/to/your_traj.txt \
  --duration-sec 40 \
  --output-json /abs/path/to/output.json \
  --output-txt /abs/path/to/output.txt \
  --dry-run
```

重点看这几行输出：

- `duration control enabled`
- `final=... pts`
- `planned_trajectory=...`
- `saved json: ...`
- `saved txt : ...`

如果 `planned_trajectory=40.000s` 且没有报错，说明至少预处理链路是通的。

### 4.2 再去掉 `--dry-run`

确认无误后，再执行真正回放：

```bash
python3 script/experiments/smooth_reteach_trajectory.py \
  --input /abs/path/to/your_traj.txt \
  --duration-sec 40 \
  --output-json /abs/path/to/output.json \
  --output-txt /abs/path/to/output.txt
```

## 5. 新轨迹复用模板

以后新增一条轨迹，直接按下面模板改文件名即可：

```bash
cd /home/chenlvping/1_DM_work/vio_eval

python3 script/experiments/smooth_reteach_trajectory.py \
  --input /home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/<name>.txt \
  --duration-sec 40 \
  --output-json /home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/smoothed/<name>.json \
  --output-txt /home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/smoothed/<name>.txt
```

第一次建议先加 `--dry-run`。

## 6. 什么时候改 `--duration-sec`

默认建议先用：

- `--duration-sec 40`

原因：

- 这是 `6_18.txt` 已成功复现的参数
- 对应自动计算出的 `dt` 约为 `5.668 ms`
- 也落在脚本给 `movej_canfd` 的建议区间 `<= 10 ms` 内

如果觉得动作偏慢或偏快，再改总时长：

- 更快：减小 `--duration-sec`
- 更慢：增大 `--duration-sec`

但要注意：

- `duration-sec` 太大，`dt` 会变大，回放会更稀疏
- `duration-sec` 太小，动作会更激进，真实机械臂负担更大

## 7. 遇到混入非轨迹行时怎么清洗

如果文件里混有 `{"gripper":1}` 一类状态行，可以先抽出仅保留 `point` 的版本：

```bash
mkdir -p /home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/cleaned

python3 - <<'PY'
from pathlib import Path

src = Path("/home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/6_23_2.txt")
dst = Path("/home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/cleaned/6_23_2_points_only.txt")

with src.open() as fi, dst.open("w") as fo:
    for line in fi:
        if '"point"' in line:
            fo.write(line)

print(dst)
PY
```

然后对清洗后的文件执行标准模板：

```bash
python3 script/experiments/smooth_reteach_trajectory.py \
  --input /home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/cleaned/6_23_2_points_only.txt \
  --duration-sec 40 \
  --output-json /home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/smoothed/6_23_2.json \
  --output-txt /home/chenlvping/1_DM_work/vio_eval/data/SJQ_data/smoothed/6_23_2.txt
```

## 8. 回放前最少检查项

- 机械臂控制器 IP 和端口是否仍是脚本默认值
- 当前轨迹首点是否安全，机械臂移动到首点不会碰撞
- 输出路径是否写成绝对路径
- 新轨迹是否只含 `point` 行
- 第一次先跑 `--dry-run`

## 9. 常见故障

### 9.1 `missing 'point' field`

原因：输入文件里混入了非轨迹行。  
处理：先做清洗，只保留 `{"point":[...]}`。

### 9.2 输出文件跑到了 `script/data/`

原因：使用了脚本默认输出名或等价相对路径。  
处理：把 `--output-json` 和 `--output-txt` 改成绝对路径。

### 9.3 觉得动作太慢

先不要急着改很多平滑参数。  
优先只改：

```bash
--duration-sec 40
```

例如改成：

```bash
--duration-sec 30
```

### 9.4 只想验证预处理，不想真动机械臂

加：

```bash
--dry-run
```

## 10. 推荐默认结论

对后续 `SJQ_data` 同类轨迹，默认按下面顺序处理：

1. 检查输入是否全是 `point`
2. 先用 `6_18` 模板跑 `--dry-run`
3. 输出路径统一使用绝对路径
4. 默认先用 `--duration-sec 40`
5. 确认结果正常后，再执行真实回放
