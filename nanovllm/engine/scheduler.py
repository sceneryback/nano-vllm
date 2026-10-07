from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """在 token 预算和 KV block 预算内组织 prefill/decode 批次。

    调度优先级是 waiting prefill 优先；只有本轮没有 prefill 时才组成 decode batch。
    """

    def __init__(self, config: Config):
        # 最大并发序列数，就是多少个 prompt，默认 512
        self.max_num_seqs = config.max_num_seqs
        # 单次 prefill 的 token 总预算；max_num_seqs 则限制请求条数，两者独立约束。
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        # 一个 kv cache block 包含的 token 数，默认 256
        self.block_size = config.kvcache_block_size
        # 管理 kv cache block 的分配和释放，相当于显存的映射
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    # 所有队列为空
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """新请求先进入 FIFO waiting 队列。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """返回本轮序列和阶段标志；True 为 prefill，False 为 decode。"""
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 没有足够的空闲 block 可以分配
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # 没有足够的空闲 block 可以分配
                if num_cached_blocks == -1:
                    break
                # 需要新分配的 token 数
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 当前 seq 需要新分配的 token 数
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 只允许 batch 中第一条序列做 chunked prefill。否则后续大请求会占用
            # 剩余预算却不能完成，增加状态组合复杂度。
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode 每条活跃序列本轮只处理一个 token，因此主要受 max_num_seqs 限制。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 缺块时从 running 尾部抢占低优先级序列；若只剩自己则抢占自己，
                # 释放 KV 后回 waiting，未来重新 prefill。
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 保持原顺序放回队首；下一轮仍优先推进这批活跃请求。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

# 抢占 seq，释放其占用的 block，放入 waiting 队列的最前面
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """提交本轮 KV 进度、追加采样 token，并回收已结束请求。"""
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill 尚未覆盖完整 prompt；本轮 logits 不用于生成。
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
