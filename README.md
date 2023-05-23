# 启真医学大模型

QiZhen: Tuning LLM Model With Chinese Medical KnowledgeBase and Instructions


本项目利用[启真医学知识库](http://www.mk-base.com)构建的中文医学指令数据集，并基于此在LLaMA-7B模型上进行指令精调，大幅提高了模型在中文医疗场景下效果，首先针对药品知识问答发布了评测数据集，后续计划优化疾病、手术、检验等方面的问答效果，并针对医患问答、病历自动生成等应用展开拓展。

## 更新记录及计划

### 更新

[2023/05/23] 开源启真医学大模型体验版，旨在提高医学领域药品知识问答的准确性；

### 计划

1. 继续在该指令集训练，进一步提高模型效果，并将模型进行开源；
2. 构造并开源药品知识问答数据集，并对开源模型进行评测，发布评测结果；
3. 构造疾病知识指令集，使用该指令集训练新模型，并将模型进行开源；
4. 使用启真医学知识库文本对LLaMA继续进行预训练，增强LLaMA在中文医疗领域的自然语言处理的基础能力；

## 指令数据集构建

目前大多数开源的ChatLLM项目使用的是其他模型（如：ChatGPT）生成的指令数据，其不可避免的存在数据幻想的问题，而数据幻想在医疗领域是无法接受的。因此，本项目为了提高医疗领域的知识问答的准确性，使用如下方式构造指令数据集：

1. 启真医学知识库收录的真实医患知识问答数据（疾病、药品、检查检验、手术、预后、食物等），共计`560K`条指令数据；
2. 药品知识数据：在启真医学知识库的药品文本知识基础上，通过对半结构化数据设置特定的问题模板（如：“{药品}的适应病症是什么？”）构造指令数据集，共计`180K`条指令数据；

## 训练细节

本项目基于[Chinese-LLaMA-Plus-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca)进行指令微调，该项目在7张A800(80G)上进行训练，本次开源的是LoRA权重为训练过程中的第`3500 steps`（训练了23h）	。

## 模型下载

|      模型       | 指令数据集 |      Base Model       |                           LoRA下载                           |
| :-------------: | :--------: | :-------------------: | :----------------------------------------------------------: |
| checkpoint-3500 |    740K    | Chinese-LLaMA-Plus-7B | [百度网盘](链接: https://pan.baidu.com/s/1KQIF-dUsL7Nrj8UeNuFUiw?pwd=ivgg 提取码: ivgg <br/>--来自百度网盘超级会员v4的分享) |

## A Quick Start

1. 环境安装；

```bash
# pip install -r requirements.txt
```

2. 获取Chinese-LLaMA-Plus-7B，详情见[这里](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2)；

3. 将模型下载并放在lora目录下；

4. 执行scripts/merge_llama_plus.sh 脚本；

```
sh scripts/merge_llama_plus.sh
```

5. 启动demo；

```
python gradio_demo.py
```

## 实验评测





## 项目组织

项目的核心开发者包括：[姚畅](https://person.zju.edu.cn/changyao)，[王贵宣](https://github.com/DendiHust)等。

## 致谢

本项目基于以下开源项目二次开发，在此对相关项目和研究开发人员表示感谢。

- [LLaMA](https://github.com/facebookresearch/llama)
- [Standford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [中文LLaMA & Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca)



## License及免责声明

### License

详见[LICENSE](./LICENSE)

### 免责声明

**本项目相关资源仅供学术研究之用，严禁用于商业用途。** 使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。



## 引用说明

coming soon



