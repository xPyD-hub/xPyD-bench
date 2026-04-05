# xPyD-bench 使用指南

> Benchmarking tool for [xPyD-proxy](https://github.com/xPyD-hub/xPyD-proxy) — measure latency, throughput, and PD-disaggregated inference performance.

## 安装

```bash
pip install xpyd-bench

# 可选依赖
pip install xpyd-bench[tokenizer]   # tiktoken 精确 token 计数
pip install xpyd-bench[http2]       # HTTP/2 支持
pip install xpyd-bench[dev]         # 开发 & 测试
```

从源码安装：

```bash
git clone https://github.com/xPyD-hub/xPyD-bench.git
cd xPyD-bench
pip install -e ".[dev]"
```

## 核心命令

### xpyd-bench

主入口，运行 benchmark。

```bash
xpyd-bench [OPTIONS]
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url` | `http://127.0.0.1:8000` | 目标服务地址 |
| `--model` | *(自动检测)* | 模型名称，省略时从 server 获取 |
| `--endpoint` | `/v1/completions` | API 端点路径 |
| `--num-prompts` | `1000` | 总请求数 |
| `--request-rate` | `inf` | 每秒请求数，`inf` 表示全部并发发送 |
| `--max-concurrency` | *(无限制)* | 最大并发请求数 |
| `--input-len` | `256` | 输入 prompt token 长度 |
| `--output-len` | `128` | 最大输出 token 数 |
| `--stream` / `--no-stream` | *(自动)* | 启用/禁用流式响应 |
| `--duration` | *(无)* | 固定运行时长（秒），到时间自动停止 |
| `--dataset-name` | `random` | 数据集类型：`random` / `synthetic` |
| `--dataset-path` | *(无)* | 自定义数据集文件路径（.jsonl/.json/.csv） |
| `--seed` | `0` | 随机种子 |
| `--burstiness` | `1.0` | 突发系数（1.0 = Poisson 分布） |
| `--repeat` | `1` | 重复运行次数 |
| `--repeat-delay` | `0` | 重复运行间隔（秒） |
| `--output` / `-o` | *(stdout)* | 结果输出文件路径 |
| `--backend` | `openai` | 后端类型 |
| `--backend-plugin` | *(无)* | 自定义后端插件模块路径 |

#### 采样参数

| 参数 | 说明 |
|------|------|
| `--temperature` | 采样温度 |
| `--top-p` | Nucleus sampling |
| `--frequency-penalty` | 频率惩罚 |
| `--presence-penalty` | 存在惩罚 |
| `--stop` | 停止序列 |

### 其他子命令

```bash
xpyd-bench compare    # 对比多次 benchmark 结果
xpyd-bench profile    # 性能剖析模式
xpyd-bench replay     # 回放录制的请求
xpyd-bench config-dump      # 导出当前配置
xpyd-bench config-validate  # 验证配置文件
xpyd-dummy             # 启动 dummy server 用于测试
```

## 典型使用场景

### 1. 单机测试

对一个本地运行的 vLLM / xPyD 实例跑 benchmark：

```bash
xpyd-bench \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B \
  --num-prompts 500 \
  --max-concurrency 32 \
  --input-len 512 \
  --output-len 256 \
  --stream \
  -o results.json
```

### 2. PD 分离（Prefill-Decode Disaggregation）测试

通过 xpyd-proxy 路由到独立的 prefill / decode 节点：

```bash
# 1) 启动 prefill 节点 (xpyd-sim 或真实 vLLM)
xpyd-sim --role prefill --port 8100

# 2) 启动 decode 节点
xpyd-sim --role decode --port 8200

# 3) 启动 proxy
xpyd-proxy --prefill http://localhost:8100 --decode http://localhost:8200 --port 8080

# 4) 跑 benchmark（对 proxy）
xpyd-bench \
  --base-url http://localhost:8080 \
  --num-prompts 1000 \
  --request-rate 50 \
  --stream \
  -o pd_results.json
```

### 3. 多模型对比

使用内置的 multi-model comparison 模式：

```bash
# 对比两个模型
xpyd-bench \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B \
  --num-prompts 200 \
  -o model_a.json

xpyd-bench \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-14B \
  --num-prompts 200 \
  -o model_b.json

# 对比结果
xpyd-bench compare model_a.json model_b.json
```

### 4. 持续时间模式

不限请求数，跑固定时长：

```bash
xpyd-bench \
  --base-url http://localhost:8000 \
  --duration 300 \
  --request-rate 20 \
  --stream
```

### 5. 使用 dummy server 快速验证

无需真实模型，启动模拟服务器：

```bash
# 终端 1：启动 dummy server
xpyd-dummy --port 8000

# 终端 2：跑 benchmark
xpyd-bench --base-url http://localhost:8000 --num-prompts 100
```

## 结果解读

### 核心指标

| 指标 | 含义 |
|------|------|
| **TTFT** (Time To First Token) | 从发送请求到收到第一个 token 的时间。反映 prefill 阶段延迟。 |
| **TPOT** (Time Per Output Token) | 每个输出 token 的平均生成时间。反映 decode 阶段速度。 |
| **TPS** (Tokens Per Second) | 每秒生成的 token 数（单请求 / 聚合）。 |
| **Throughput** | 总吞吐量：请求数/秒（req/s）和 token 数/秒（tok/s）。 |
| **Error Rate** | 失败请求占比。 |

### 百分位指标

| 指标 | 含义 |
|------|------|
| **P50** | 中位数，50% 的请求低于此值 |
| **P90** | 90% 的请求低于此值 |
| **P99** | 99% 的请求低于此值，反映长尾延迟 |
| **Mean** | 算术平均值 |
| **Std** | 标准差，反映延迟稳定性 |

### 如何判断结果好坏

- **TTFT < 200ms**：prefill 性能良好（7B 模型、512 token 输入）
- **TPOT < 30ms**：decode 速度正常
- **P99/P50 < 3x**：延迟分布健康，无严重长尾
- **Error Rate = 0%**：服务稳定
- **Throughput**：随并发增长应接近线性，达到饱和后趋于平稳

### 结果文件

输出 JSON 包含：

```json
{
  "config": { ... },
  "results": {
    "total_requests": 1000,
    "successful_requests": 998,
    "failed_requests": 2,
    "total_duration_s": 45.2,
    "requests_per_second": 22.1,
    "tokens_per_second": 2834,
    "ttft_ms": { "mean": 152, "p50": 140, "p90": 210, "p99": 380 },
    "tpot_ms": { "mean": 22, "p50": 20, "p90": 28, "p99": 45 },
    "latency_ms": { "mean": 2950, "p50": 2800, "p90": 3500, "p99": 4200 }
  }
}
```

## 与 xpyd-sim / xpyd-proxy 配合使用

### 架构

```
Client (xpyd-bench)
        │
        ▼
   xpyd-proxy (路由层)
     ┌──┴──┐
     ▼     ▼
  Prefill  Decode
  (xpyd-sim / vLLM)
```

### 完整示例

参见 [`scripts/run_benchmark.sh`](../scripts/run_benchmark.sh) 了解一键启动脚本。

```bash
# 手动步骤
pip install xpyd-sim xpyd-proxy xpyd-bench

# 启动 sim 节点
xpyd-sim --role prefill --port 8100 &
xpyd-sim --role decode  --port 8200 &

# 启动 proxy
xpyd-proxy \
  --prefill http://localhost:8100 \
  --decode  http://localhost:8200 \
  --port 8080 &

# 等待服务就绪
sleep 3

# 运行 benchmark
xpyd-bench \
  --base-url http://localhost:8080 \
  --num-prompts 500 \
  --request-rate 30 \
  --max-concurrency 64 \
  --input-len 256 \
  --output-len 128 \
  --stream \
  -o benchmark_results.json

echo "Results saved to benchmark_results.json"
```

## 高级功能

- **Checkpoint & Resume** (`--checkpoint`)：长时间 benchmark 中断后可恢复
- **Benchmark Fingerprint** (`--fingerprint`)：唯一标识 benchmark 配置，方便对比
- **Configuration Inheritance** (`--extends`)：配置文件继承
- **Rolling Window Metrics**：实时滚动窗口统计
- **Baseline Registry**：注册基线结果，后续自动对比回归
- **Speculative Decoding Metrics**：投机解码相关指标
- **Prefix Caching Impact**：前缀缓存效果分析
- **Adaptive Timeout**：根据观察到的延迟自动调整超时
- **Multimodal Vision Benchmark**：支持 vision 模型测试
- **SLA Validation**：定义 SLA 规则，自动判断是否达标
- **Distributed Benchmark**：多节点协调分布式压测
