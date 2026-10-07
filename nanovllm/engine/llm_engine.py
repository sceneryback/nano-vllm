import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """面向用户的同步生成引擎，也是 TP rank 0 的调度控制面。"""

    def __init__(self, model, **kwargs):
        # 只把 Config 声明过的关键字传入，允许上层保留其他 API 参数而不报错。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 所有 Sequence 共享同一引擎的块大小，避免每个实例重复保存。
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 为每个 TP 初始化一个 ModelRunner，通常一个 TP 对应一张卡
        # 从 1号卡开始，0号卡单独拉起
        for i in range(1, config.tensor_parallel_size):
            # 创建一个跨进程的事件对象，用于进程间同步（一个进程 set，另一个进程 wait）
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            # fork 或 spawn 一个新进程，执行 ModelRunner(config, i, event)
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 在主进程构造；构造期间 warmup 并回填 num_kvcache_blocks，
        # 因此 Scheduler 必须在它之后创建。
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        """通知 TP workers 退出，释放共享内存和 NCCL process group。"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """把文本或预分词 token ids 包装为 Sequence 并加入 waiting 队列。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 将 prompt 初始化为 seq，计算 
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

# 核心逻辑
    def step(self):
        """执行一次 prefill/decode 调度、模型运行和状态提交。"""
        # 调度一批 prompt
        seqs, is_prefill = self.scheduler.schedule()
        # 用正负号复用一个返回值：正数统计 prefill tokens，负数统计 decode tokens。
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 在 prefill 阶段，是各 seq 的首 token；在 decode 阶段是各 seq 的新生成 token
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """阻塞运行全部请求，并按原始 seq_id 顺序返回解码结果。"""
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            # 请求添加到 scheduler
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            # 单调递增的纳秒级高精度计时器
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 请求可能乱序完成，按递增 seq_id 恢复调用方传入顺序。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
