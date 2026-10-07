import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    """最小文本生成示例：应用 chat template 后批量生成两个回答。"""
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 指最终的 LM head 怎么选取输出新的 token 以及最多生成多少 token
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    # apply_chat_template 只生成字符串；真正的 tokenize 在 LLMEngine.add_request 中。
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
