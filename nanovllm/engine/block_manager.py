from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        # 从 0 开始的序号
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 管理逻辑 kvcache 块，相当于 cpu 侧的元数据
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # num_blocks 在 ModelRunner 中初始化，当前剩余显存能分配多少个 block
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash 到 block_id 的映射，如果某个 block 的 hash 等于另一个 block，说明这两个 block 前序的所有 tokens 也相同
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 不同 block 的 hash 可能相同
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        # 当前 seq 理论上要分配的 block，实际上没有这么多，因为可以用缓存
        num_new_blocks = seq.num_blocks
        # 遍历前 n-1 个 block
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            # 计算当前 block 的 hash，其中包含了前一个 block 的 hash
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 之前没有过相同的 block
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            # 命中的 block
            num_cached_blocks += 1
            # 该 block id 已经使用，则需要新分配的数量减一
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
                # 没有足够的空闲 block 分配
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        # 返回可以缓存的 block 数量
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        # cached blocks 也可能来自其他 prompt
        # cached blocks 只可能来自序列的前几个，某个 block 缓存命中，意味着前边的所有 block 都命中了，因为是链式 hash
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 其他 block 新分配
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
