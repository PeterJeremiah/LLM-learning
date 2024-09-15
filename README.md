# LLM-learning

### 教程安利：如果你也想了解大模型，可以去了解一下哦！[大模型实战营](https://github.com/InternLM/Tutorial)!

参考：

- ([self-llm from github](https://github.com/datawhalechina/self-llm?tab=readme-ov-file))
- ([Tramac from zhihu](https://zhuanlan.zhihu.com/p/660101812))
- ([BimAnt Blog](http://www.bimant.com/blog/local-llm-and-vlm-performance-test/))

## 1.1 LLM和LVM

### 1.1.1 LLM

### 1.1.2 VLM

## 1.2 LLM模型测评

最近初步了解大模型时，经常被各种 LLMs 模型搞的眼花缭乱，想要挑选一个性能还不错在本地部署大模型，所以这里收集了一些被广泛认可且目前还比较活跃的LLMs评测榜单，用于跟踪最新的模型和效果。

### 1.2.1 LMSYS

![image-20240910213331685](C:\Users\yangmy\AppData\Roaming\Typora\typora-user-images\image-20240910213331685.png)

**简介**：`LMSYS` 推出的 `Chatbot Arena` 是一个以众包方式进行匿名的` LLMs` 基准平台，主要包含以下3个基准：

- Chatbot Arena：一个[大语言模型](https://zhida.zhihu.com/search?q=大语言模型&zhida_source=entity&is_preview=1)基准平台，目前已有 90K+ 用户的投票数据，采用 Elo 评级方法进行计算得到的结果。
- [MT-Bench](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2306.05685)：一个多轮问题对话 benchmark，并且使用 [GPT-4](https://zhida.zhihu.com/search?q=GPT-4&zhida_source=entity&is_preview=1) 的结果作为标准进行评分。
- [MMLU](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2009.03300) (5-shot)：一个常用于衡量模型在多任务准确性的` benchmark`，主要涵盖了基础数学、美国历史、计算机科学、法律等57项任务。

**Leaderboard**：[Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

GitHub：[https://github.com/lm-sys/FastC](https://github.com/lm-sys/FastChat)

### 1.2.2 Open LLM Leaderboard

**简介**：由` Hugging Face` 发布，**主要针对英文**的评测榜单，旨在跟踪、排名和评估开源的` LLMs`

- [AI2 Reasoning Challenge](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1803.05457) (25-shot)：主要涵盖了一些小学科学问题。
- [HellaSwag](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1905.07830) (10-shot)：[常识推理数据集](https://zhida.zhihu.com/search?q=常识推理数据集&zhida_source=entity&is_preview=1)
- [MMLU](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2009.03300) (5-shot)：一个常用于衡量模型在多任务准确性的 `benchmark`，主要涵盖了基础数学、美国历史、计算机科学、法律等57项任务。
- [TruthfulQA](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2109.07958) (0-shot)：一个用于测试模型谎言倾向的` benchmark`。

**Leaderboard**：[Chatbot Arena Leaderboard](https://link.zhihu.com/?target=https%3A//huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

GitHub：[https://github.com/lm-sys/FastC](https://link.zhihu.com/?target=https%3A//github.com/lm-sys/FastChat)



## 1.3 本地部署大模型

### 1.3.1 [FastApi 部署调用](https://github.com/datawhalechina/self-llm/blob/master/models/Llama3_1/01-Llama3_1-8B-Instruct%20FastApi%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md)

* ConnectionRefusedError: [WinError 10061] 由于目标计算机积极拒绝，无法连接。

### 1.3.2 Ollama 运维工具

* `Ollama`与`Llama`的关系：`Llama`是大语言模型，而`Ollama`是大语言模型（不限于`Llama`模型）便捷的管理和运维工具

* `Ollama`目前支持以下大语言模型：[library](https://ollama.com/library)

* [**Qwen-7B**](https://huggingface.co/yzsydlc/qwen2)

* [部署方法](https://blog.csdn.net/qq_30298311/article/details/139810505)

### 1.3.3 [本地 LLM & VLM 性能测评方案](http://www.bimant.com/blog/local-llm-and-vlm-performance-test/)

