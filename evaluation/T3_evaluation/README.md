# T3_evaluation：正式宿主评分

角色跟随Track1/2 evaluation：读取预测和私有GT，输出metrics.json。本版依照主办方要求直接在宿主Python运行，无评测Docker依赖。
它不是单元测试；单元测试在根目录verification。

```bash
python evaluate.py --predictions-root /path/to/docker_output --gt-root /path/to/private_gt --output-root /path/to/scores
python rank.py --reports /path/to/trusted_reports --output /path/to/leaderboard
```

valid/test使用同一脚本，只替换主文件夹路径。接口、GT manifest、四指标和并列规则详见根README和SCORING.md。
prepare_data.py用于整理现有ATM数据；run_submission.py和run_from_config.py是主办方可选的Docker运行封装，参赛者无需提交它们。
paths.example.json中的四个宿主路径可以直接编辑；容器内部/input、/output是固定接口，不是本机磁盘路径。
