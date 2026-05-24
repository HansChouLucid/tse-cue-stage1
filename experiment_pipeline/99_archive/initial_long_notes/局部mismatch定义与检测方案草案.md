# 局部 mismatch 定义与检测方案草案

## 1. 目标

本阶段要验证的不是“模型整体分数高不高”，而是：

> 在 hard condition 下，fine-grained cue 是否会在局部窗口内把目标说话人/目标内容跟干扰源混淆。

这里的关键是“局部”，所以 mismatch 必须按 chunk/window 来定义和检测，而不能只看 utterance 级别的 SV 分数。

---

## 2. 核心定义

### 2.1 局部 mismatch

对输出语音 `y` 按固定窗口切成若干 chunk `y_t`。
如果某个局部 chunk 在“身份归属”或“内容归属”上更接近干扰源而不是目标源，则记为局部 mismatch。

最基础的身份型定义：

```text
gap_id(t) = sim(verifier(y_t), verifier(target_t)) - sim(verifier(y_t), verifier(interferer_t))
```

当 `gap_id(t) < margin` 时，认为 chunk `t` 出现 identity mismatch。

内容型定义：

```text
gap_content(t) = sim(content_rep(y_t), content_rep(target_t)) - sim(content_rep(y_t), content_rep(interferer_t))
```

当 `gap_content(t) < margin` 时，认为 chunk `t` 出现 content mismatch。

### 2.2 两类 mismatch

- `similar speaker` 主要看 **identity mismatch**
- `similar content` 主要看 **content mismatch**
- 两者叠加时，看二者是否同时恶化

---

## 3. 为什么不能直接搬用 utterance-level 方法

utterance-level SV 方法通常经过 pooling，能很好回答“整段像不像目标说话人”，但不擅长发现局部异常，原因是：

1. 局部错配会被全局平均掉
2. 短 chunk 的 embedding 更噪，阈值不稳定
3. similar content 下，内容/音素相似会干扰身份判断

所以局部 mismatch 不能只靠一个全局 embedding 分数，需要重新局部化、重新校准、再配合底层信号。

---

## 4. 检测框架

建议采用“三层检测”：

### 4.1 第一层：局部 verifier

对每个 chunk 计算局部身份或内容相似度，得到：

- `chunk identity gap`
- `chunk content gap`
- `mismatch rate`
- `first mismatch position`
- `mismatch duration`

这是主检测层。

### 4.2 第二层：底层解释信号

用于解释为什么发生 mismatch，不单独作为最终判定：

- attention shift
- cross-attention concentration
- T-F interaction map 异常
- local score drop trajectory
- temporal drift

### 4.3 第三层：连续性判定

局部 mismatch 最重要的不只是“有没有”，而是：

- 从哪里开始错
- 持续多久
- 是否连续出现

因此需要加滑窗/邻接块平滑规则，避免把单个噪声 chunk 当成真正 mismatch。

---

## 5. similar speaker 的检测方案

### 5.1 主要假设

干扰源与目标源声纹相似时，模型可能把局部输出 chunk 误归为干扰者。

### 5.2 主检测指标

- `Local Speaker Gap`
- `Mismatch Rate`
- `Mismatch Duration`
- `First Mismatch Position`
- `Chunk-level Speaker Score Trajectory`

### 5.3 推荐 verifier

优先用局部 speaker verifier，而不是只用 utterance-level embedding。

建议顺序：

1. speaker encoder chunk embedding
2. chunk embedding centroid / prototype
3. 输出 chunk 与 target/interferer 的相似度差

### 5.4 辅助诊断

只做解释，不直接作为最终判定：

- attention 是否偏向干扰者活跃区
- 局部 identity score 是否突然掉下去
- 相邻 chunk 的 identity 是否不稳定

---

## 6. similar content 的检测方案

### 6.1 主要假设

干扰源与目标源在词汇、短语、音素模式上相似时，模型可能出现内容归属错误，而不一定是身份错配。

### 6.2 主检测指标

- `Local Content Attribution Gap`
- `Chunk-level Content Drift`
- `Phonetic Match Shift`
- `Content Mismatch Rate`

### 6.3 推荐 verifier

这里不能只靠 speaker verifier，因为 speaker verifier 未必能看出内容被谁带偏。

建议：

1. 用 SSL / ASR / phone-like frame representation 做局部内容表示
2. 比较 output chunk 与 target chunk、interferer chunk 的内容相似度
3. 再看身份 gap 是否同时恶化

### 6.4 关键点

`similar content` 的 mismatch 更像：

> 输出局部内容更接近干扰者的语音片段/词汇模式，而不是单纯 speaker 身份偏移。

所以这里最好同时看：

- content gap
- identity gap
- attention / alignment 是否被内容相似性吸引

---

## 7. 两类任务的区分

### 7.1 similar speaker

重点问题：

- 输出局部身份是否偏向干扰者
- verifier 是否能发现这种局部身份漂移

优先看：

- local speaker gap
- mismatch duration
- temporal identity consistency

### 7.2 similar content

重点问题：

- 输出局部内容是否被相似短语/音素诱导
- speaker verifier 单独是否不够

优先看：

- local content gap
- phonetic drift
- identity gap + content gap 联合异常

---

## 8. 建议的指标集合

### 8.1 身份类

- `Chunk Identity Gap`
- `Mismatch Rate`
- `Mismatch Duration`
- `First Mismatch Position`
- `Temporal Identity Consistency`

### 8.2 内容类

- `Chunk Content Gap`
- `Content Mismatch Rate`
- `Phonetic Drift Score`

### 8.3 解释类

- `Attention Shift Score`
- `Interaction Map Entropy`
- `Local Confidence Drop`
- `Multi-scale Verifier Score`

---

## 9. 评测协议建议

### 9.1 按条件跑

1. normal
2. similar speaker
3. similar content
4. similar speaker + similar content

### 9.2 按模型跑

1. global-only
2. fine-grained-only
3. naive fusion
4. fine-grained + diagnostic global verifier

### 9.3 按尺度跑

建议至少三种窗口：

- 0.5s
- 1.0s
- 2.0s

这样可以看 mismatch 在不同时间尺度下是否稳定。

---

## 10. 实际执行顺序

### Step 1: 定义 chunk 切分

- 固定窗口长度
- 固定 hop
- 固定滑窗平滑规则

### Step 2: 定义 verifier

- similar speaker: 局部 speaker verifier
- similar content: 局部 content verifier

### Step 3: 固定 mismatch 判定式

- 用 gap + margin
- 再加连续性判定

### Step 4: 在 smoke 集上验证

- 先看指标能不能稳定产生
- 再看和 hard condition 是否一致

### Step 5: 再跑正式实验矩阵

- normal / similar speaker / similar content / both
- global-only / fine-grained-only / fusion / verifier

---

## 11. 当前最重要的结论

1. 局部 mismatch 必须按 chunk 定义，不能只靠 utterance-level 分数。
2. similar speaker 和 similar content 要分开检测。
3. attention 可以作为底层解释信号，但不要单独当最终判定。
4. utterance-level SV 方法可以借用，但要局部化、重新校准。
5. 最终的研究重点是：

> fine-grained cue 在 hard condition 下是否会产生可检测、可解释的局部错配。

