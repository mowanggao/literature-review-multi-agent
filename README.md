
# 学术文献综述多 Agent 协作系统

基于大语言模型的智能文献综述生成系统，通过 5 个专业 Agent 协同工作，快速生成高质量的学术文献综述。

## 功能特点

- **多源文献检索**：支持 ArXiv 和 Semantic Scholar 双平台检索
- **智能信息提取**：自动提取研究问题、方法、结论和创新点
- **技术脉络梳理**：按时间顺序分析领域发展历程
- **研究缺口识别**：系统分析未解决的科学问题
- **规范格式输出**：自动生成 GB/T 7714-2015 格式参考文献

## 系统架构

```
┌─────────────────┐
│  检索 Agent     │ → 从 ArXiv/Semantic Scholar 获取文献
└────────┬────────┘
         │
┌────────▼────────┐
│  精读 Agent     │ → 提取文献关键信息
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│脉络   │ │缺口   │
│推理   │ │分析   │
│Agent  │ │Agent  │
└───┬───┘ └──┬────┘
    │         │
    └────┬────┘
         │
┌────────▼────────┐
│  格式 Agent     │ → 生成最终综述文档
└─────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 OpenAI API Key：

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### 3. 运行系统

```bash
# 默认主题
python main.py

# 自定义主题
python main.py "Federated Learning"
python main.py "Segment Anything Model"
```

## 项目结构

```
literature-review-multi-agent/
├── agents/
│   ├── __init__.py
│   ├── retrieval_agent.py    # 文献检索
│   ├── extraction_agent.py   # 信息提取
│   ├── reasoning_agent.py    # 脉络推理
│   ├── gap_agent.py          # 缺口分析
│   └── format_agent.py       # 格式校验
├── config.py                 # 配置文件
├── main.py                   # 主程序入口
├── requirements.txt          # 依赖列表
├── .env.example              # 环境变量示例
└── output/                   # 输出目录（自动创建）
```

## 使用示例

```python
from main import LiteratureReviewSystem

system = LiteratureReviewSystem()
result = system.run(
    topic="YOLO object detection",
    max_papers=15,
    years=5
)

print(result["formatted_review"])
```

## 输出说明

系统会在 `output/` 目录下生成两个文件：

1. `literature_review_{topic}_{timestamp}.md` - 完整的文献综述
2. `intermediate_{topic}_{timestamp}.json` - 中间处理结果

## 技术栈

- LangChain - 大语言模型应用框架
- OpenAI API - 大语言模型接口
- ArXiv API - 学术文献检索
- Semantic Scholar API - 学术文献检索
- Pydantic - 数据验证

## 注意事项

1. 确保 API Key 有效且有足够额度
2. 网络连接需要能访问 OpenAI API（或配置代理）
3. 首次运行可能需要较长时间（取决于文献数量）

# 学术文献综述多Agent协作系统
🔥 基于大模型 + 多智能体协作，自动完成**文献检索 → 精读提取 → 脉络推理 → 缺口分析 → 格式生成**的一站式文献综述生成系统

## 项目介绍
传统文献综述需要人工检索、阅读、梳理、总结，耗时数周。
本项目使用 **5个Agent分工协作**，自动生成高质量学术综述，大幅提升科研效率。

## 核心功能
✅ 自动检索 ArXiv 文献
✅ 智能精读提取研究问题、方法、结论、创新点
✅ 长链推理梳理技术演进脉络
✅ 自动分析研究缺口与未来方向
✅ 自动生成 GB/T 7714 参考文献格式

## 技术栈
- Python
- LangChain
- OpenAI GPT / 国产大模型
- ArXiv API

## 快速运行
```bash
pip install langchain openai arxiv python-dotenv pymupdf
979bb716ebca835f249b2ea5696f7012edc51687
