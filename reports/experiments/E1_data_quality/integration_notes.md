# PR #10 与当前 main 的集成核验

关联 Issue：#9；PR：#10。

原实验从固定 baseline-v1（41772b6）训练，原配置、结果、environment.json 和嵌入式 Notebook 保留为训练快照，不重新生成或覆盖指标。

原 PR 含基线历史提交并与当前 main 冲突。本次合入 main（437d0dc），冲突文件采用 main 版本，保留 E2/E3 和共用模型/损失接口。只在 src/data/dataset.py 恢复 E1 所需的 red_blue 提取分支，阈值和并集规则与训练快照一致，默认 red 行为不变。未改动共用 train/evaluate/infer、冻结切分、基线报告或其他成员结果（相对当前 main）。

验证结果：

- 原有 test_core.py：4 项通过；新增 test_e1_masks.py：2 项通过。
- 三张真实 OVERLAY 的 red/red_blue 输出逐像素等于训练快照。
- 冻结切分 SHA256 与云端 environment.json 一致。
- 新增脚本语法检查通过；相对 main 的改动通过 git diff --check。
- 本地没有 PyTorch，未在集成版本重跑模型训练和 GPU 推理；云端训练快照已完成训练、评估与样例生成。不能将本次集成检查写成新的完整 GPU 实验。

内部参考掩码公平对照、重复图像泄漏处理、基线环境一致性核验仍待负责人协作。上述集成不消除这些科学评估限制。
