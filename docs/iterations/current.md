# xPyD-bench — 当前迭代状态

> 更新日期：2026-04-05

## 当前里程碑：M88 — Speculative Decoding Metrics

M88 已完成，新增投机解码（Speculative Decoding）相关的 benchmark 指标支持。

## 主要功能列表

### 核心 Benchmark
- OpenAI-compatible API benchmark（completions / chat）
- 流式 & 非流式响应支持
- Poisson / 突发请求调度（`--request-rate`, `--burstiness`）
- 自定义数据集加载（random / synthetic / JSONL / CSV）
- Token bucket 限流
- 多后端插件架构（openai / 自定义 plugin）

### 高级测试模式
- **Multi-model comparison** (M75) — 多模型对比
- **Streaming vs non-streaming overhead** (M76) — 流式开销分析
- **Multimodal vision benchmark** (M77) — 视觉模型支持
- **Multi-turn conversation** — 多轮对话测试
- **Chain benchmark** — 请求链路测试
- **Sweep mode** — 参数扫描
- **Distributed benchmark** — 多节点协调压测

### 指标 & 分析
- TTFT / TPOT / TPS / throughput / error rate
- 自定义百分位（P50 / P90 / P99 / 用户自定义）(M73)
- Rolling window metrics (M81) — 滚动窗口实时统计
- Confidence intervals — 置信区间
- Latency breakdown — 延迟分解
- Workload distribution statistics (M78)
- Speculative decoding metrics (M88)
- Prefix caching impact analysis (M87)
- Anomaly detection — 异常检测

### 可靠性 & 运维
- Benchmark checkpointing & resume (M74)
- Benchmark fingerprinting (M72) — 配置唯一标识
- Baseline registry (M82) — 基线注册与回归对比
- Error threshold abort (M83) — 错误率超限自动终止
- Request deduplication & idempotency (M85)
- Adaptive timeout auto-tuning (M86)
- Git metadata capture (M79) — 结果绑定 git 版本
- Configuration inheritance via `extends` (M80)

### 报告 & 集成
- Rich CLI 输出
- JSON / HTML / JUnit XML 报告
- Prometheus 指标导出
- WebSocket 实时指标推送
- Webhook 通知
- OTLP trace 导出
- Heatmap 可视化

### 工具
- `xpyd-dummy` — 模拟服务器，无需真实模型即可测试
- `xpyd-bench compare` — 结果对比
- `xpyd-bench profile` — 性能剖析
- `xpyd-bench replay` — 请求回放
- `xpyd-bench config-dump / config-validate` — 配置管理

## 已知限制

1. **不支持 gRPC 后端** — 目前仅支持 HTTP/1.1 和 HTTP/2（需安装 `h2`）
2. **Token 计数依赖 tiktoken** — 需额外安装 `xpyd-bench[tokenizer]`，否则使用粗略估算
3. **Distributed mode 无自动发现** — 需手动指定 worker 地址
4. **无内置图表** — HTML 报告仅包含表格，需配合外部工具（Grafana + Prometheus exporter）可视化
5. **License 待定** — 尚未确定开源协议

## 下一步计划

- **M89+**：参见 [ROADMAP.md](../../ROADMAP.md) 获取完整路线图
- 优先方向：
  - GPU 利用率关联分析
  - A/B 测试框架（自动统计显著性检验）
  - 更丰富的 HTML 报告（内嵌图表）
  - gRPC 后端支持
  - 自动发现集群节点
