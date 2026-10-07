from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """一个物理 KV block 的 CPU 元数据；真正的 K/V tensor 位于 GPU。"""

    def __init__(self, block_id):
        # 从 0 开始的序号
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """完整块计算完成后，记录其链式 hash 和用于防碰撞校验的 token。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """把可回收块交给新请求；新 K/V 写完前它还不是有效 prefix cache。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 管理逻辑 kvcache 块，相当于 cpu 侧的元数据
class BlockManager:
    """管理逻辑序列到物理 KV block 的映射和 prefix cache 生命周期。"""

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
        """计算链式 block hash，使当前 hash 同时依赖全部前缀块。

        例如 ``h0=H(block0)``、``h1=H(h0, block1)``，因此 block1 内容
        相同但 block0 不同的两条请求不会错误共享 KV。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """取一个可回收块并使旧缓存索引失效。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 不同 block 的 hash 可能相同，有全局索引指向了这个 block，应该先删除，因为这个 block 要分配新的 kv 了
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """标记为可回收，但保留 hash 和显存内容供未来 prefix 命中。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """返回连续命中的 prefix block 数；空间不足返回 -1。

        空闲但缓存命中的块仍占用一个 free slot；正在使用的命中块可共享，才会
        让 ``num_new_blocks`` 减一。
        """
        h = -1
        num_cached_blocks = 0
        # 当前 seq 理论上要分配的 block，实际上没有这么多，因为可以用缓存
        num_new_blocks = seq.num_blocks
        # 最后一块不复用：它可能未填满，而且至少重算末块才能得到末 token logits。
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
        """复用缓存前缀并为剩余逻辑块预留物理块，建立完整 block_table。"""
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
            # num_cached_blocks 来自 hash_to_block_id，在 deallocate 中不会释放，但 used_block_ids 中可能释放了，所以此时可能不命中
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
        """释放序列引用；共享块仅在最后一个引用离开时进入 free 队列。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # append_token 已先让长度 +1；余数为 1 表示刚跨入一个新逻辑块。
        # bool 在数值比较中等价于 0/1：需要新块时至少要有 1 个 free block。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """decode 刚跨块边界时，为新 token 分配下一个物理块。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """把本轮新填满的块登记为可复用 prefix cache。

        整除自然排除不完整尾块。例：cached=256、scheduled=300、block=256，
        ``start=1, end=2``，只登记刚完成的逻辑 block 1。
        """
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
