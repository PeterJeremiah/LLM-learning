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



### 2 Tools

| Notebook                                                   | Description                                                  | Notebook                                                     |
| ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 🧐 [LLM AutoEval](https://github.com/mlabonne/llm-autoeval) | Automatically evaluate your LLMs using RunPod                | [![Open In Colab](https://github.com/mlabonne/llm-course/raw/main/img/colab.svg)](https://colab.research.google.com/drive/1Igs3WZuXAIv9X0vwqiE90QlEPys8e8Oa?usp=sharing) |
| 🥱 LazyMergekit                                             | Easily merge models using MergeKit in one click.             | [![Open In Colab](https://github.com/mlabonne/llm-course/raw/main/img/colab.svg)](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing) |
| 🦎 LazyAxolotl                                              | Fine-tune models in the cloud using Axolotl in one click.    | [![Open In Colab](https://github.com/mlabonne/llm-course/raw/main/img/colab.svg)](https://colab.research.google.com/drive/1TsDKNo2riwVmU55gjuBgB1AXVtRRfRHW?usp=sharing) |
| ⚡ AutoQuant                                                | Quantize LLMs in GGUF, GPTQ, EXL2, AWQ, and HQQ formats in one click. | [![Open In Colab](https://github.com/mlabonne/llm-course/raw/main/img/colab.svg)](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing) |
| 🌳 Model Family Tree                                        | Visualize the family tree of merged models.                  | [![Open In Colab](https://github.com/mlabonne/llm-course/raw/main/img/colab.svg)](https://colab.research.google.com/drive/1s2eQlolcI1VGgDhqWIANfkfKvcKrMyNr?usp=sharing) |
| 🚀 ZeroSpace                                                | Automatically create a Gradio chat interface using a free ZeroGPU. | [![Open In Colab](https://github.com/mlabonne/llm-course/raw/main/img/colab.svg)](https://colab.research.google.com/drive/1LcVUW5wsJTO2NGmozjji5CkC--646LgC) |

### 

## 3 自注意力机制(Self-Attention Mechanism)

>自注意力(Self-Attention)机制也称为内部注意力(Intra-Attention)，是一种特殊的注意力机制。自注意力机制作为一种新型的网络结构被广泛应用于自然语言处理与计算机视觉等任务中。

* ([Self-Attention Mechanism](https://0809zheng.github.io/2020/04/24/self-attention.html))

### 3.1 Attention and Self-Attention

**注意力机制**(attention mechanism)最早是在序列到序列模型中提出的，用于解决机器翻译任务。

输入$ \{x1,x2,...,xj\} $ 转换序列 $\{y1,y2,...yi\}$。
$$
y_i = \sum_{j}^{}w_{ij}x_j
$$
引入约束$\sum_{j}^{}w_{ij}=1$。$w_{ij}$ 不是学习得到，而是输入计算得到。

注意机制，其主要原则是在一个句子中形成一个词和每个其他词之间的联系，本质上，捕捉一个参数中所有词之间的关系。

从这个意义上说，所有的单词都与所有其他单词有联系，我们称之为这个单词/输入/标记的自我注意权重。

**自注意力机制**则不相同，权重参数是由输入决定的，即使是同一个模型，对于不同的输入也会有不同的权重参数。

### 3.2 CNN, RNN, Self-Attention

卷积神经网络事实上只能获得局部信息，需要通过**堆叠**更多层数来增大感受野；循环神经网络需要通过**递归**获得全局信息，因此一般采用双向形式；自注意力机制能够直接获得**全局信息**。

卷积神经网络事实上只能获得局部信息，需要通过堆叠更多层数来增大感受野；循环神经网络需要通过递归获得全局信息，因此一般采用双向形式；自注意力机制能够直接获得全局信息。

### 3.3 Self-Attention

* ([Self-Attention by hand](https://twitter.com/ProfTomYeh/status/1797249951325434315))

* ([Exploring architectures- Transformers II](https://mathblog.vercel.app/blog/transformers2/))

Positional Encoding：在 Transformer 体系结构中，并行计算受到青睐，因此每个自注意权重都是同时计算的，因此我们必须以某种方式将单词的位置信息编码到模型中。这一步称为位置编码。

Residual Connections（剩余连接）：解决梯度消失问题。

Layer Normalization（分层归一化）：保持在检查的梯度，同时也缓解了网络内的梯度流。它也有助于收敛的权重参数，其最佳值，其中的损失是在一个理想的水平。

Loss fuction：交叉熵损失被认为介于预测的概率分布和真实的概率分布之间，我们必须将矩阵通过一个柔性最大激活函数，但是应用于每一行，以便将它们约束在一个分布中，这个分布将与真实的概率分布 T 进行比较。

Backpropagation（反向传播）：将通过损失函数向后传播，以便计算每个权重参数相对于损失函数的梯度**W**Q, **W**K, **W**V 和 **W**。
