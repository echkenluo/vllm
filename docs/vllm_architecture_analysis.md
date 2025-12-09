# vLLM 架构与实现深度解析

本文档对 vLLM 代码仓库进行了全面的架构分析，涵盖核心原理、数据结构和关键算法。

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [入口点与请求流程](#3-入口点与请求流程)
4. [核心引擎](#4-核心引擎)
5. [调度器系统](#5-调度器系统)
6. [PagedAttention 与 KV Cache 管理](#6-pagedattention-与-kv-cache-管理)
7. [模型执行器与 Worker](#7-模型执行器与-worker)
8. [分布式执行与并行策略](#8-分布式执行与并行策略)
9. [采样与输出处理](#9-采样与输出处理)
10. [配置系统](#10-配置系统)
11. [CUDA 内核优化](#11-cuda-内核优化)
12. [V1 新架构](#12-v1-新架构)
13. [核心数据结构汇总](#13-核心数据结构汇总)

---

## 1. 项目概述

vLLM 是一个高吞吐、内存高效的大语言模型推理和服务引擎。其核心创新是 **PagedAttention**，通过内存虚拟化技术实现高效的 KV Cache 管理。

### 核心特性

- **PagedAttention**: 将 KV Cache 分页管理，支持非连续内存存储
- **连续批处理 (Continuous Batching)**: 动态调度请求，最大化 GPU 利用率
- **前缀缓存 (Prefix Caching)**: 跨请求共享相同前缀的 KV Cache
- **投机解码 (Speculative Decoding)**: 使用 draft 模型加速生成
- **多种并行策略**: 张量并行 (TP)、流水线并行 (PP)、数据并行 (DP)、专家并行 (EP)

### 目录结构

```
vllm/
├── entrypoints/          # 用户入口点 (LLM 类, CLI, OpenAI API)
├── engine/               # 引擎核心逻辑
├── v1/                   # V1 新架构实现
│   ├── engine/           # V1 引擎
│   ├── core/             # 调度器和 KV Cache 管理
│   ├── worker/           # Worker 和 ModelRunner
│   └── executor/         # 执行器
├── model_executor/       # 模型执行和加载
│   ├── models/           # 模型实现 (Llama, GPT, etc.)
│   └── layers/           # 可复用层 (注意力, 线性, 量化)
├── attention/            # 注意力后端
├── distributed/          # 分布式通信
├── config/               # 配置系统
└── csrc/                 # C++/CUDA 内核
```

---

## 2. 整体架构

### 类层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户入口 (Entrypoints)                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  LLM 类     │  │ OpenAI API 服务器 │  │    CLI       │  │
│  │ (离线推理)   │  │   (在线服务)      │  │  (命令行)    │  │
│  └──────┬──────┘  └────────┬─────────┘  └───────────────┘  │
└─────────┼──────────────────┼───────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    引擎层 (Engine Layer)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LLMEngine / AsyncLLM                    │   │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │InputProcessor│ │EngineCore   │ │OutputProcessor│  │   │
│  │  └─────────────┘ └──────────────┘ └──────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   调度层 (Scheduler Layer)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Scheduler                         │   │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │RequestQueue │ │KVCacheManager│ │ BlockPool    │  │   │
│  │  └─────────────┘ └──────────────┘ └──────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   执行层 (Executor Layer)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │    Executor (UniProc / MultiProc / Ray)              │   │
│  │           │                                          │   │
│  │    ┌──────┴──────┬──────────────┬──────────────┐    │   │
│  │    ▼             ▼              ▼              ▼    │   │
│  │ Worker 0     Worker 1      Worker 2      Worker N   │   │
│  │    │             │              │              │    │   │
│  │ ModelRunner  ModelRunner  ModelRunner  ModelRunner  │   │
│  │    │             │              │              │    │   │
│  │  Model        Model         Model         Model     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 请求生命周期

```
1. 请求接收 → 2. 输入处理 → 3. 调度 → 4. 执行 → 5. 采样 → 6. 输出处理

详细流程:
┌──────────┐    ┌───────────────┐    ┌──────────────┐
│ 用户请求  │ -> │ InputProcessor │ -> │EngineCoreReq │
└──────────┘    │ - 分词         │    │ - token_ids  │
               │ - 多模态处理    │    │ - 采样参数    │
               └───────────────┘    └──────┬───────┘
                                          │
                                          ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ SchedulerOut │ <- │  Scheduler   │ <- │   请求队列    │
│ - 调度决策    │    │ - 选择请求   │    │ - 等待队列   │
│ - 块分配     │    │ - 分配KVCache│    │ - 运行队列   │
└──────┬───────┘    └──────────────┘    └──────────────┘
       │
       ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ModelRunnerOut│ <- │  Executor    │ <- │   Workers    │
│ - 采样token  │    │ - 分发任务   │    │ - 前向传播   │
│ - logprobs   │    │ - 收集结果   │    │ - 注意力计算 │
└──────┬───────┘    └──────────────┘    └──────────────┘
       │
       ▼
┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│RequestOutput │ <- │OutputProcessor│ <- │EngineCoreOut │
│ - 生成文本   │    │ - 解码        │    │ - new_tokens │
│ - finish原因 │    │ - 停止检测    │    │ - finish_rsn │
└──────────────┘    └───────────────┘    └──────────────┘
```

---

## 3. 入口点与请求流程

### 3.1 LLM 类 (离线推理)

**文件位置**: `vllm/entrypoints/llm.py`

```python
from vllm import LLM, SamplingParams

# 初始化
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 生成
outputs = llm.generate(
    prompts=["Hello, my name is"],
    sampling_params=SamplingParams(temperature=0.8, max_tokens=100)
)
```

**内部流程**:
```
LLM.__init__()
  ├─ EngineArgs.create_engine_config()  # 创建配置
  └─ LLMEngine.from_engine_args()       # 创建引擎

LLM.generate()
  ├─ _validate_and_add_requests()       # 添加请求
  │   └─ InputProcessor.process_inputs()
  └─ _run_engine()                      # 运行引擎循环
      └─ while has_unfinished:
          └─ llm_engine.step()
```

### 3.2 OpenAI 兼容 API 服务器

**文件位置**: `vllm/entrypoints/openai/api_server.py`

启动命令:
```bash
vllm serve <model> --port 8000
```

**架构**:
```
FastAPI Server
    │
    ├─ /v1/chat/completions  → OpenAIServingChat
    ├─ /v1/completions       → OpenAIServingCompletion
    ├─ /v1/embeddings        → OpenAIServingEmbedding
    └─ /health               → HealthCheck

OpenAIServing*
    │
    └─ engine_client.generate()  → AsyncGenerator[RequestOutput]
```

### 3.3 CLI 命令

```bash
vllm serve      # 启动 API 服务器
vllm bench      # 性能基准测试
vllm run-batch  # 批量推理
vllm chat       # 交互式对话
```

---

## 4. 核心引擎

### 4.1 LLMEngine (V1)

**文件位置**: `vllm/v1/engine/llm_engine.py`

LLMEngine 是同步推理的核心组件，职责包括:
- 请求管理和路由
- 输入输出处理协调
- 日志和统计收集

**关键属性**:
```python
class LLMEngine:
    input_processor: InputProcessor    # 输入预处理
    output_processor: OutputProcessor  # 输出后处理
    engine_core: EngineCoreClient      # 引擎核心客户端
    logger_manager: StatLoggerManager  # 统计日志
```

**核心方法**:

```python
def add_request(self, request_id, prompt, params):
    """添加新请求到引擎"""
    # 1. 输入处理: 分词、多模态预处理
    engine_core_req = self.input_processor.process_inputs(prompt, params)

    # 2. 注册输出状态
    self.output_processor.add_request(request_id, params)

    # 3. 添加到调度器
    self.engine_core.add_request(engine_core_req)

def step(self) -> List[RequestOutput]:
    """执行一次推理迭代"""
    # 1. 获取引擎输出
    outputs = self.engine_core.get_output()

    # 2. 处理输出 (解码、停止检测)
    return self.output_processor.process_outputs(outputs)
```

### 4.2 AsyncLLM (异步引擎)

**文件位置**: `vllm/v1/engine/async_llm.py`

用于在线服务的异步包装器:

```python
class AsyncLLM:
    async def generate(self, prompt, params) -> AsyncGenerator[RequestOutput]:
        """异步生成，支持流式输出"""
        # 启动后台输出处理器
        self._run_output_handler()

        # 添加请求
        await self.add_request(request_id, prompt, params)

        # 流式返回结果
        async for output in self.output_collector:
            yield output
```

### 4.3 EngineCore

**文件位置**: `vllm/v1/engine/core.py`

EngineCore 是实际的推理引擎，管理调度和执行:

```python
class EngineCore:
    scheduler: Scheduler           # 调度器
    model_executor: Executor       # 模型执行器

    def step(self) -> EngineCoreOutputs:
        """核心推理循环"""
        # 1. 调度决策
        scheduler_output = self.scheduler.schedule()

        # 2. 模型执行
        model_output = self.model_executor.execute_model(scheduler_output)

        # 3. 采样 (如果需要)
        sampled = self.model_executor.sample_tokens(model_output)

        # 4. 更新调度器状态
        return self.scheduler.update_from_output(sampled)
```

**多进程模式**:

当使用多进程时，EngineCore 运行在独立进程中，通过 ZMQ 通信:

```
主进程 (API Server)           子进程 (EngineCore)
     │                              │
     │ ── ZMQ: add_request ──────> │
     │                              │ scheduler.schedule()
     │                              │ executor.execute_model()
     │ <── ZMQ: EngineCoreOutput ── │
     │                              │
```

---

## 5. 调度器系统

### 5.1 调度器架构

**文件位置**: `vllm/v1/core/sched/scheduler.py`

vLLM V1 使用**统一的 token 预算调度**，不区分 prefill 和 decode 阶段:

```python
class Scheduler:
    waiting_queue: RequestQueue       # 等待队列
    running: dict[str, Request]       # 运行中请求
    kv_cache_manager: KVCacheManager  # KV Cache 管理

    def schedule(self) -> SchedulerOutput:
        """调度算法核心"""
        scheduled = {}
        token_budget = self.max_num_batched_tokens

        # 1. 调度运行中的请求
        for req in self.running.values():
            tokens_needed = req.num_tokens - req.num_computed_tokens
            if token_budget >= tokens_needed:
                scheduled[req.id] = tokens_needed
                token_budget -= tokens_needed

        # 2. 调度等待中的请求
        while not self.waiting_queue.empty() and token_budget > 0:
            req = self.waiting_queue.peek()
            if self._can_allocate(req):
                scheduled[req.id] = min(req.num_tokens, token_budget)
                self.waiting_queue.pop()

        return SchedulerOutput(num_scheduled_tokens=scheduled)
```

### 5.2 请求状态机

```
                    ┌──────────────────────┐
                    │                      │
                    ▼                      │
┌─────────┐    ┌─────────┐    ┌─────────┐  │  ┌─────────────┐
│ WAITING │ -> │ RUNNING │ -> │FINISHED │  │  │  PREEMPTED  │
└─────────┘    └────┬────┘    └─────────┘  │  └──────┬──────┘
                    │                      │         │
                    │    (KV Cache 不足)   │         │
                    └──────────────────────┴─────────┘

状态说明:
- WAITING: 在等待队列中
- WAITING_FOR_FSM: 等待结构化输出 FSM 编译
- WAITING_FOR_REMOTE_KVS: 等待远程 KV 传输
- RUNNING: 正在执行
- PREEMPTED: 被抢占，等待重新调度
- FINISHED_*: 完成 (STOPPED/LENGTH_CAPPED/ABORTED)
```

### 5.3 请求数据结构

**文件位置**: `vllm/v1/request.py`

```python
@dataclass
class Request:
    request_id: str

    # Token 追踪
    num_tokens: int              # 总 token 数 (prompt + output)
    num_computed_tokens: int     # 已计算的 token 数
    num_prompt_tokens: int       # prompt token 数
    num_output_tokens: int       # 已生成的 output token 数

    # 前缀缓存
    num_cached_tokens: int       # 缓存命中的 token 数

    # 状态
    status: RequestStatus

    # Token IDs (只追加列表)
    all_token_ids: ConstantList[int]
```

### 5.4 调度输出

```python
@dataclass
class SchedulerOutput:
    # 每个请求调度的 token 数
    num_scheduled_tokens: dict[str, int]

    # 新请求数据
    scheduled_new_reqs: list[NewRequestData]

    # 缓存请求数据
    scheduled_cached_reqs: list[CachedRequestData]

    # 投机解码 token
    scheduled_spec_decode_tokens: dict[str, list[int]]

    # 统计
    total_num_scheduled_tokens: int
```

---

## 6. PagedAttention 与 KV Cache 管理

### 6.1 PagedAttention 原理

PagedAttention 是 vLLM 的核心创新，将 KV Cache 分成固定大小的块进行管理:

```
传统方式 (连续内存):
┌─────────────────────────────────────────┐
│ Request 1 KV Cache (预分配最大长度)      │
├─────────────────────────────────────────┤
│ Request 2 KV Cache (预分配最大长度)      │
├─────────────────────────────────────────┤
│          大量浪费的内存空间              │
└─────────────────────────────────────────┘

PagedAttention (分页内存):
┌──────┬──────┬──────┬──────┬──────┬──────┐
│Blk 0 │Blk 1 │Blk 2 │Blk 3 │Blk 4 │Blk 5 │
│ R1   │ R1   │ R2   │ R1   │ R2   │ Free │
└──────┴──────┴──────┴──────┴──────┴──────┘

Block Table (逻辑到物理映射):
Request 1: [0, 1, 3]     # 逻辑块 0,1,2 -> 物理块 0,1,3
Request 2: [2, 4]        # 逻辑块 0,1 -> 物理块 2,4
```

### 6.2 KV Cache 块结构

**文件位置**: `vllm/v1/core/kv_cache_utils.py`

```python
@dataclass
class KVCacheBlock:
    block_id: int           # 物理块 ID [0, num_gpu_blocks)
    ref_cnt: int = 0        # 引用计数 (支持共享)
    block_hash: BlockHash   # 内容哈希 (前缀缓存)

    # 双向链表指针 (LRU 驱逐)
    prev_free_block: KVCacheBlock
    next_free_block: KVCacheBlock
```

### 6.3 块池管理

**文件位置**: `vllm/v1/core/block_pool.py`

```python
class BlockPool:
    blocks: list[KVCacheBlock]                # 所有块
    free_block_queue: FreeKVCacheBlockQueue   # LRU 空闲队列
    cached_block_hash_to_block: BlockHashToBlockMap  # 前缀缓存

    def allocate(self) -> KVCacheBlock:
        """分配一个块"""
        block = self.free_block_queue.popleft()  # O(1)
        if block.block_hash:
            self._evict_cached_block(block)
        block.ref_cnt = 1
        return block

    def free(self, block: KVCacheBlock):
        """释放一个块"""
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self.free_block_queue.append(block)  # O(1)
```

### 6.4 KV Cache 管理器

**文件位置**: `vllm/v1/core/kv_cache_manager.py`

```python
class KVCacheManager:
    block_pool: BlockPool
    req_to_blocks: dict[str, list[KVCacheBlock]]

    def allocate_slots(self, request, num_tokens) -> list[KVCacheBlock]:
        """为请求分配 KV Cache 槽位"""
        # 1. 检查前缀缓存
        cached_blocks = self._get_cached_blocks(request)

        # 2. 分配新块
        num_new_blocks = ceil((num_tokens - cached_tokens) / block_size)
        new_blocks = self.block_pool.allocate_n(num_new_blocks)

        # 3. 更新块表
        self.req_to_blocks[request.id].extend(new_blocks)

        return new_blocks

    def free(self, request_id: str):
        """释放请求的所有块"""
        blocks = self.req_to_blocks.pop(request_id)
        # 逆序释放以保持 LRU 顺序
        self.block_pool.free_blocks(reversed(blocks))
```

### 6.5 前缀缓存算法

```python
def hash_block_tokens(parent_hash, token_ids, extra_keys) -> BlockHash:
    """计算块的内容哈希"""
    # 哈希 = hash(父块哈希, token_ids, 额外键)
    # 额外键包括: 多模态特征, LoRA 名称, cache_salt
    return BlockHash(hash_func((parent_hash, tuple(token_ids), extra_keys)))

def get_cached_blocks(request) -> list[KVCacheBlock]:
    """查找缓存命中的块"""
    cached = []
    parent_hash = None

    for i in range(0, len(request.token_ids), block_size):
        block_tokens = request.token_ids[i:i+block_size]
        if len(block_tokens) < block_size:
            break  # 不完整的块不缓存

        block_hash = hash_block_tokens(parent_hash, block_tokens, ...)
        cached_block = self.block_pool.get_cached(block_hash)

        if cached_block is None:
            break  # 缓存未命中

        cached.append(cached_block)
        parent_hash = block_hash

    return cached
```

### 6.6 Slot Mapping 计算

**文件位置**: `vllm/v1/worker/block_table.py`

```python
def compute_slot_mapping(req_indices, positions, block_table, block_size):
    """计算 token 到物理内存位置的映射"""
    # 1. 计算块索引
    block_indices = req_indices * max_blocks_per_req + positions // block_size

    # 2. 获取物理块号
    block_numbers = block_table[block_indices]

    # 3. 计算块内偏移
    block_offsets = positions % block_size

    # 4. 计算最终槽位
    slot_mapping = block_numbers * block_size + block_offsets

    return slot_mapping

# 示例 (block_size=16):
# Request 0 的 token 位置 5 -> 逻辑块 0, 偏移 5
# 块表: Request 0 的逻辑块 0 -> 物理块 7
# slot_mapping = 7 * 16 + 5 = 117
```

---

## 7. 模型执行器与 Worker

### 7.1 执行器层次

```
Executor (抽象基类)
    │
    ├── UniProcExecutor      # 单进程执行
    │
    ├── MultiprocExecutor    # 多进程执行 (Python multiprocessing)
    │   └── WorkerProc       # Worker 进程封装
    │
    └── RayDistributedExecutor  # Ray 分布式执行
        └── Ray Actors
```

### 7.2 Worker 实现

**文件位置**: `vllm/v1/worker/gpu_worker.py`

```python
class Worker:
    """管理单个 GPU 的 Worker"""

    def __init__(self, vllm_config, rank, local_rank):
        # 设置 CUDA 设备
        self.device = f"cuda:{local_rank}"
        torch.cuda.set_device(self.device)

        # 初始化分布式环境
        init_distributed_environment(rank, world_size)

        # 创建 ModelRunner
        self.model_runner = GPUModelRunner(vllm_config)

    def load_model(self):
        """加载模型权重"""
        self.model_runner.load_model()

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        """执行模型前向传播"""
        return self.model_runner.execute_model(scheduler_output)
```

### 7.3 ModelRunner 实现

**文件位置**: `vllm/v1/worker/gpu_model_runner.py`

```python
class GPUModelRunner:
    """GPU 上的模型执行引擎"""

    def __init__(self, vllm_config):
        self.model = None
        self.sampler = Sampler()
        self.request_states = RequestState()
        self.cudagraph_manager = CudaGraphManager()

    def load_model(self):
        """加载并初始化模型"""
        loader = get_model_loader(self.load_config)
        self.model = loader.load_model(self.vllm_config)

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        """执行一次推理"""
        # 1. 准备输入
        input_batch = self.prepare_inputs(scheduler_output)

        # 2. 更新块表
        self.block_table.update(scheduler_output)

        # 3. 执行模型
        if self.cudagraph_enabled:
            hidden_states = self.cudagraph_manager.run(input_batch)
        else:
            hidden_states = self.model.forward(
                input_ids=input_batch.token_ids,
                positions=input_batch.positions,
                kv_caches=self.kv_caches,
                attn_metadata=input_batch.attn_metadata,
            )

        return hidden_states

    def sample_tokens(self, hidden_states) -> ModelRunnerOutput:
        """采样生成 token"""
        logits = self.model.compute_logits(hidden_states)
        sampled = self.sampler.sample(logits, self.sampling_metadata)
        return ModelRunnerOutput(sampled_token_ids=sampled)
```

### 7.4 模型加载与权重分片

**文件位置**: `vllm/model_executor/model_loader/`

```python
# 权重加载流程
def load_model(vllm_config):
    # 1. 获取模型加载器
    loader = DefaultModelLoader(load_config)

    # 2. 创建模型实例
    model = create_model_instance(vllm_config)

    # 3. 加载权重
    for name, weight in loader.weights_iterator():
        # 对于张量并行，只加载本 rank 的分片
        param = model.get_parameter(name)
        param.weight_loader(param, weight, shard_id=tp_rank)

    return model

# 权重分片示例 (TP=2)
# 原始权重: [hidden_size, hidden_size]
# Rank 0 加载: [:, :hidden_size//2]
# Rank 1 加载: [:, hidden_size//2:]
```

---

## 8. 分布式执行与并行策略

### 8.1 并行策略概述

```
┌─────────────────────────────────────────────────────────────┐
│                      并行策略                                │
├─────────────────┬─────────────────┬─────────────────────────┤
│   张量并行 (TP)  │  流水线并行 (PP) │      数据并行 (DP)       │
├─────────────────┼─────────────────┼─────────────────────────┤
│ 单层内权重切分   │ 跨层切分        │ 请求级别并行             │
│ All-Reduce 通信 │ P2P 通信        │ 独立处理不同请求         │
│ 减少显存占用    │ 减少单 GPU 层数 │ 提高总吞吐量             │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### 8.2 张量并行 (TP)

**文件位置**: `vllm/distributed/parallel_state.py`

```python
# TP 组创建
# World size = 8, TP = 4, PP = 2
# TP 组: [0,1,2,3], [4,5,6,7]

def initialize_tensor_parallel(world_size, tp_size):
    all_ranks = torch.arange(world_size)
    tp_groups = all_ranks.view(-1, tp_size).unbind(0)
    # tp_groups = [[0,1,2,3], [4,5,6,7]]

    for group in tp_groups:
        dist.new_group(group.tolist())
```

**TP 层实现**:

```python
class ColumnParallelLinear(nn.Module):
    """列并行线性层 - 输出维度切分"""

    def __init__(self, in_features, out_features):
        tp_size = get_tensor_parallel_size()
        self.out_features_per_partition = out_features // tp_size
        self.weight = Parameter(torch.empty(
            self.out_features_per_partition, in_features
        ))

    def forward(self, x):
        # 输入复制到所有 rank
        # 每个 rank 计算部分输出
        return F.linear(x, self.weight)

class RowParallelLinear(nn.Module):
    """行并行线性层 - 输入维度切分"""

    def __init__(self, in_features, out_features):
        tp_size = get_tensor_parallel_size()
        self.in_features_per_partition = in_features // tp_size
        self.weight = Parameter(torch.empty(
            out_features, self.in_features_per_partition
        ))

    def forward(self, x):
        # 输入已分片
        local_out = F.linear(x, self.weight)
        # All-Reduce 合并
        return tensor_model_parallel_all_reduce(local_out)
```

### 8.3 流水线并行 (PP)

```python
# PP 组创建
# World size = 8, TP = 2, PP = 4
# PP 组: [0,2,4,6], [1,3,5,7]

def initialize_pipeline_parallel(world_size, pp_size, tp_size):
    all_ranks = torch.arange(world_size).view(-1, pp_size)
    pp_groups = all_ranks.transpose(0, 1).unbind(0)
    # 每组包含不同 PP 阶段的 rank

# 层分配
# PP Rank 0: layers[0:8]   (第一阶段)
# PP Rank 1: layers[8:16]  (第二阶段)
# PP Rank 2: layers[16:24] (第三阶段)
# PP Rank 3: layers[24:32] (第四阶段)
```

### 8.4 专家并行 (EP)

用于 MoE (Mixture of Experts) 模型:

```python
# EP 组创建
# 每个专家分布在不同 GPU 上
# 使用 All-to-All 通信重新分配 token

class MoELayer:
    def forward(self, x):
        # 1. 路由决策
        router_logits = self.gate(x)
        topk_weights, topk_indices = topk(router_logits, k=2)

        # 2. All-to-All 发送 token 到对应专家
        dispatched = all_to_all(x, topk_indices)

        # 3. 本地专家计算
        expert_out = self.experts[local_expert_id](dispatched)

        # 4. All-to-All 收集结果
        return all_to_all(expert_out, reverse=True)
```

### 8.5 执行器 RPC 通信

**文件位置**: `vllm/v1/executor/multiproc_executor.py`

```python
class MultiprocExecutor:
    def __init__(self, vllm_config):
        # 创建 worker 进程
        self.workers = []
        for rank in range(world_size):
            proc = WorkerProc(rank, vllm_config)
            proc.start()
            self.workers.append(proc)

        # 消息队列
        self.input_queue = MessageQueue()   # 广播输入
        self.output_queues = [MessageQueue() for _ in range(world_size)]

    def collective_rpc(self, method, args):
        """向所有 worker 发送 RPC 请求"""
        # 广播请求
        self.input_queue.put((method, args))

        # 只从指定 rank 收集响应 (通常是 rank 0)
        return self.output_queues[0].get()
```

---

## 9. 采样与输出处理

### 9.1 SamplingParams 参数

**文件位置**: `vllm/sampling_params.py`

```python
@dataclass
class SamplingParams:
    # 温度控制 (随机性)
    temperature: float = 1.0      # 0=贪婪, >1=更随机
    top_k: int = 0                # Top-K 采样 (0=禁用)
    top_p: float = 1.0            # Nucleus 采样 (1.0=禁用)
    min_p: float = 0.0            # 最小概率阈值

    # 惩罚参数 (抑制重复)
    presence_penalty: float = 0.0     # 出现惩罚
    frequency_penalty: float = 0.0    # 频率惩罚
    repetition_penalty: float = 1.0   # 重复惩罚

    # 输出控制
    n: int = 1                    # 生成序列数
    max_tokens: int = 16          # 最大生成 token 数
    min_tokens: int = 0           # 最小生成 token 数

    # 停止条件
    stop: list[str] = None        # 停止字符串
    stop_token_ids: list[int] = None  # 停止 token ID

    # Logprobs
    logprobs: int = None          # 返回 top-k logprobs
    prompt_logprobs: int = None   # 返回 prompt logprobs
```

### 9.2 采样算法流程

**文件位置**: `vllm/v1/sample/sampler.py`

```python
def sample(logits, sampling_metadata) -> SampledTokens:
    """采样流程"""

    # 1. Logprobs 准备 (在处理前保存原始 logits)
    if need_logprobs:
        raw_logprobs = log_softmax(logits.clone())

    # 2. 应用非 argmax 不变的处理器
    logits = apply_min_tokens(logits, ...)
    logits = apply_logit_bias(logits, ...)
    logits = apply_allowed_tokens(logits, ...)
    logits = apply_bad_words(logits, ...)

    # 3. 应用惩罚
    logits = apply_repetition_penalty(logits, prompt_tokens, output_tokens)
    logits -= frequency_penalty * token_counts
    logits -= presence_penalty * token_mask

    # 4. 贪婪采样路径
    if all_greedy:
        return argmax(logits)

    # 5. 温度缩放
    logits = logits / temperature

    # 6. 应用 argmax 不变的处理器
    logits = apply_min_p(logits, min_p)

    # 7. Top-K/Top-P 过滤
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    # 8. 随机采样 (Gumbel-max trick)
    probs = softmax(logits)
    q = torch.empty_like(probs).exponential_()
    sampled = (probs / q).argmax(dim=-1)

    return sampled
```

### 9.3 Top-P (Nucleus) 采样

```python
def apply_top_p(logits, p):
    """Nucleus 采样 - 保留累积概率 <= p 的 token"""
    # 按概率降序排序
    sorted_logits, sorted_indices = logits.sort(descending=True)
    sorted_probs = softmax(sorted_logits)

    # 计算累积概率
    cumsum_probs = sorted_probs.cumsum(dim=-1)

    # 找到超过阈值的位置
    mask = cumsum_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False  # 至少保留一个 token

    # 屏蔽低概率 token
    sorted_logits[mask] = float('-inf')

    # 恢复原始顺序
    return sorted_logits.gather(-1, sorted_indices.argsort())
```

### 9.4 输出处理与解码

**文件位置**: `vllm/v1/engine/output_processor.py`

```python
class OutputProcessor:
    def process_outputs(self, engine_outputs) -> list[RequestOutput]:
        """处理引擎输出"""
        results = []

        for output in engine_outputs:
            request_state = self.get_state(output.request_id)

            # 1. 增量解码
            new_text = self.detokenizer.decode(
                output.new_token_ids,
                skip_special_tokens=True
            )

            # 2. 停止字符串检测
            stop_found = self._check_stop_strings(new_text, request_state)

            # 3. 构建输出
            completion = CompletionOutput(
                text=new_text,
                token_ids=output.new_token_ids,
                finish_reason=output.finish_reason,
                logprobs=output.new_logprobs,
            )

            results.append(RequestOutput(
                request_id=output.request_id,
                outputs=[completion],
                finished=output.finish_reason is not None,
            ))

        return results
```

---

## 10. 配置系统

### 10.1 配置层次

**文件位置**: `vllm/config/`

```
VllmConfig (主配置)
    │
    ├── ModelConfig          # 模型配置
    │   ├── model            # 模型名称/路径
    │   ├── dtype            # 数据类型
    │   ├── max_model_len    # 最大上下文长度
    │   └── quantization     # 量化方法
    │
    ├── CacheConfig          # KV Cache 配置
    │   ├── block_size       # 块大小 (16, 32, 64...)
    │   ├── gpu_memory_utilization  # GPU 内存利用率
    │   └── enable_prefix_caching   # 前缀缓存开关
    │
    ├── ParallelConfig       # 并行配置
    │   ├── tensor_parallel_size    # TP 大小
    │   ├── pipeline_parallel_size  # PP 大小
    │   └── data_parallel_size      # DP 大小
    │
    ├── SchedulerConfig      # 调度配置
    │   ├── max_num_seqs     # 最大并发序列数
    │   ├── max_num_batched_tokens  # 每次迭代最大 token 数
    │   └── enable_chunked_prefill  # 分块 prefill
    │
    ├── DeviceConfig         # 设备配置
    │
    ├── LoadConfig           # 加载配置
    │
    ├── AttentionConfig      # 注意力配置
    │
    ├── LoRAConfig           # LoRA 配置
    │
    ├── SpeculativeConfig    # 投机解码配置
    │
    └── CompilationConfig    # 编译配置
        ├── cudagraph_mode   # CUDA Graph 模式
        └── optimization_level  # 优化级别 (O0-O3)
```

### 10.2 优化级别

```python
# 优化级别定义
O0 = "O0"  # 无优化 (调试模式)
    # - 禁用 torch.compile
    # - 禁用 CUDA graphs
    # - 禁用算子融合

O1 = "O1"  # 快速启动
    # - Dynamo + Inductor 编译
    # - Piecewise CUDA graphs
    # - 基础融合 (RMSNorm, 激活)

O2 = "O2"  # 完全优化 (默认)
    # - Full + Piecewise CUDA graphs
    # - 高级融合 passes

O3 = "O3"  # 最大优化
    # - 当前与 O2 相同
    # - 预留未来优化
```

### 10.3 配置创建流程

**文件位置**: `vllm/engine/arg_utils.py`

```python
def create_engine_config(args) -> VllmConfig:
    """从命令行参数创建配置"""

    # 1. 创建模型配置
    model_config = ModelConfig(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )

    # 2. 创建缓存配置
    cache_config = CacheConfig(
        block_size=args.block_size or infer_block_size(),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 3. 创建并行配置
    parallel_config = ParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )

    # 4. 组装主配置
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        ...
    )
```

---

## 11. CUDA 内核优化

### 11.1 PagedAttention 内核

**文件位置**: `csrc/attention/`

```cpp
// PagedAttention V1 - 单分区
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* out,              // [num_seqs, num_heads, head_size]
    const scalar_t* q,          // [num_seqs, num_heads, head_size]
    const cache_t* k_cache,     // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* v_cache,     // [num_blocks, num_kv_heads, head_size, block_size]
    const int* block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* seq_lens,        // [num_seqs]
    ...
) {
    // Grid: (num_heads, num_seqs, 1)
    // 每个 thread block 处理一个 (head, seq) 对

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int seq_len = seq_lens[seq_idx];

    // 遍历 KV cache 块
    float qk_max = -FLT_MAX;
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // 通过块表获取物理块位置
        int physical_block = block_tables[seq_idx * max_blocks + block_idx];

        // 计算 Q·K^T
        for (int token_idx = 0; token_idx < BLOCK_SIZE; token_idx++) {
            float qk = dot_product(q, k_cache[physical_block][token_idx]);
            qk_max = max(qk_max, qk);
            logits[token_idx] = qk;
        }
    }

    // Softmax
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        logits[i] = exp(logits[i] - qk_max);
        sum += logits[i];
    }

    // 计算输出
    for (int i = 0; i < seq_len; i++) {
        out += logits[i] / sum * v_cache[i];
    }
}

// PagedAttention V2 - 多分区 + 归约
// 将长序列分成多个分区并行计算，然后归约
template <...>
__global__ void paged_attention_v2_kernel(...) {
    // Grid: (num_heads, num_seqs, max_num_partitions)
    // 每个分区处理 PARTITION_SIZE=512 个 token

    const int partition_idx = blockIdx.z;
    // ... 计算分区内的 attention

    // 保存分区结果
    exp_sums[partition_idx] = partition_sum;
    max_logits[partition_idx] = partition_max;
    tmp_out[partition_idx] = partition_out;
}

__global__ void paged_attention_v2_reduce_kernel(...) {
    // 合并所有分区的结果
    // 使用 log-sum-exp trick 保持数值稳定性
}
```

### 11.2 激活函数内核

```cpp
// SiLU * x (SwiGLU 门控)
template <typename scalar_t>
__global__ void silu_and_mul_kernel(
    scalar_t* out,
    const scalar_t* input,
    const int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d) {
        scalar_t x = input[idx];
        scalar_t gate = input[idx + d];
        out[idx] = x * gate / (1.0f + exp(-gate));  // x * silu(gate)
    }
}
```

### 11.3 量化内核

```cpp
// FP8 动态量化
template <typename scalar_t>
__global__ void dynamic_per_token_quant_kernel(
    const scalar_t* input,      // [num_tokens, hidden_size]
    int8_t* output,             // [num_tokens, hidden_size]
    float* scales               // [num_tokens]
) {
    // 每个 warp 处理一个 token
    int token_idx = blockIdx.x;

    // 1. 找到最大绝对值
    float max_val = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        max_val = max(max_val, abs(input[token_idx * hidden_size + i]));
    }
    max_val = warp_reduce_max(max_val);

    // 2. 计算缩放因子
    float scale = max_val / 127.0f;
    if (threadIdx.x == 0) {
        scales[token_idx] = scale;
    }

    // 3. 量化
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = input[token_idx * hidden_size + i];
        output[token_idx * hidden_size + i] = round(val / scale);
    }
}
```

### 11.4 注意力后端

| 后端 | 特点 | 适用场景 |
|------|------|----------|
| FlashAttention v2/v3 | 官方实现，稳定 | 通用场景 |
| FlashInfer | TRTLLM 优化，支持 FP4 | 高性能推理 |
| Triton | 可移植，易于修改 | 研究和定制 |
| PagedAttention | vLLM 原生，分页 KV Cache | 默认后端 |

---

## 12. V1 新架构

### 12.1 V1 vs V0 对比

| 方面 | V0 | V1 |
|------|----|----|
| **调度设计** | 阶段分离 (prefill/decode) | 统一 token 预算 |
| **Token 分配** | 按阶段固定 | 按请求动态 |
| **Chunked Prefill** | 可选启用 | 默认启用 |
| **KV Cache Swap** | 支持 GPU-CPU 交换 | 已移除 |
| **Logprobs 模式** | 仅后处理 | 原始模式 (可配置) |
| **Per-Request 处理器** | 支持 | 仅全局处理器 |

### 12.2 V1 核心改进

**1. 统一调度器**:
```python
# V0: 分阶段调度
if phase == PREFILL:
    schedule_prefill_requests()
elif phase == DECODE:
    schedule_decode_requests()

# V1: 统一 token 预算调度
for request in all_requests:
    tokens_needed = request.num_tokens - request.num_computed
    if budget >= tokens_needed:
        schedule(request, tokens_needed)
        budget -= tokens_needed
```

**2. 分块 Prefill 默认启用**:
- 长 prompt 自动分块处理
- 减少首 token 延迟 (TTFT)
- 更好的 GPU 利用率

**3. 异步调度**:
```python
# 支持批次队列，减少流水线气泡
class AsyncScheduler:
    def step_with_batch_queue(self):
        # 异步调度新批次
        future = self.executor.execute_model(scheduler_output, non_block=True)
        self.batch_queue.append(future)

        # 从队列获取完成的结果
        if len(self.batch_queue) >= self.max_concurrent:
            return self.batch_queue.popleft().result()
```

### 12.3 V1 目录结构

```
vllm/v1/
├── engine/
│   ├── core.py              # EngineCore (1,456 行)
│   ├── llm_engine.py        # LLMEngine 包装
│   ├── async_llm.py         # 异步 LLM
│   ├── input_processor.py   # 输入处理
│   ├── output_processor.py  # 输出处理
│   └── detokenizer.py       # 解码器
│
├── core/
│   ├── sched/
│   │   ├── scheduler.py     # 调度器 (1,751 行)
│   │   ├── async_scheduler.py
│   │   └── request_queue.py
│   ├── kv_cache_manager.py  # KV Cache 管理
│   ├── block_pool.py        # 块池
│   └── kv_cache_coordinator.py
│
├── worker/
│   ├── gpu_worker.py        # GPU Worker
│   ├── gpu_model_runner.py  # ModelRunner (1,006 行)
│   └── block_table.py       # 块表管理
│
├── executor/
│   ├── abstract.py          # 执行器抽象
│   ├── multiproc_executor.py
│   ├── ray_executor.py
│   └── uniproc_executor.py
│
├── attention/
│   └── backends/            # 注意力后端
│       ├── flash_attn.py
│       ├── flashinfer.py
│       └── ...
│
└── sample/
    ├── sampler.py           # 采样器
    └── ops/                  # 采样算子
```

---

## 13. 核心数据结构汇总

### 13.1 请求相关

```python
# 引擎核心请求 (msgspec.Struct 高效序列化)
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int]
    mm_features: list[MultiModalFeature]  # 多模态特征
    sampling_params: SamplingParams
    arrival_time: float
    lora_request: LoRARequest | None
    priority: int

# 引擎核心输出
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    new_logprobs: LogprobsLists | None
    finish_reason: FinishReason | None  # STOP, LENGTH, ABORT
    stop_reason: int | str | None
    num_cached_tokens: int

# 用户可见输出
class RequestOutput:
    request_id: str
    outputs: list[CompletionOutput]
    finished: bool
    prompt_token_ids: list[int]

class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    logprobs: SampleLogprobs | None
    finish_reason: str | None
```

### 13.2 调度相关

```python
# 调度器输出
class SchedulerOutput:
    num_scheduled_tokens: dict[str, int]  # {request_id: num_tokens}
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: list[CachedRequestData]
    total_num_scheduled_tokens: int
    preempted_req_ids: set[str]
    finished_req_ids: set[str]

# 模型运行器输出
class ModelRunnerOutput:
    sampled_token_ids: torch.Tensor
    logprobs: LogprobsTensors | None
    req_id_to_index: dict[str, int]
```

### 13.3 KV Cache 相关

```python
# KV Cache 块
@dataclass
class KVCacheBlock:
    block_id: int           # 物理块 ID
    ref_cnt: int            # 引用计数
    block_hash: BlockHash   # 内容哈希
    prev_free_block: Self   # LRU 链表
    next_free_block: Self

# 块表
class BlockTable:
    block_table: Tensor     # [num_reqs, max_blocks_per_req]
    slot_mapping: Tensor    # [total_tokens]
```

### 13.4 配置相关

```python
# 主配置
@dataclass
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    device_config: DeviceConfig
    load_config: LoadConfig
    attention_config: AttentionConfig
    lora_config: LoRAConfig | None
    speculative_config: SpeculativeConfig | None
    compilation_config: CompilationConfig
    optimization_level: OptimizationLevel
```

---

## 附录 A: 关键文件位置

| 组件 | 文件路径 | 行数 |
|------|----------|------|
| LLM 入口 | `vllm/entrypoints/llm.py` | ~800 |
| LLMEngine | `vllm/v1/engine/llm_engine.py` | ~400 |
| EngineCore | `vllm/v1/engine/core.py` | ~1,456 |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | ~1,751 |
| KV Cache Manager | `vllm/v1/core/kv_cache_manager.py` | ~800 |
| Block Pool | `vllm/v1/core/block_pool.py` | ~500 |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | ~600 |
| GPU ModelRunner | `vllm/v1/worker/gpu_model_runner.py` | ~1,006 |
| Sampler | `vllm/v1/sample/sampler.py` | ~600 |
| PagedAttention V1 | `csrc/attention/paged_attention_v1.cu` | ~186 |
| PagedAttention V2 | `csrc/attention/paged_attention_v2.cu` | ~196 |
| 配置系统 | `vllm/config/*.py` | ~300KB |

## 附录 B: 常用命令

```bash
# 启动 API 服务器
vllm serve meta-llama/Llama-2-7b-hf --port 8000

# 启动带张量并行
vllm serve meta-llama/Llama-2-70b-hf --tensor-parallel-size 4

# 启用量化
vllm serve TheBloke/Llama-2-7b-AWQ --quantization awq

# 运行基准测试
vllm bench throughput --model meta-llama/Llama-2-7b-hf

# 批量推理
vllm run-batch --model meta-llama/Llama-2-7b-hf --input prompts.jsonl
```

---

*文档生成时间: 2025-12*
*基于 vLLM 代码仓库 main 分支*
