import os
import arxiv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv

# ================= 1. 初始化配置 =================
load_dotenv()

# 这里以 OpenAI 为例，如果你用国内模型（如智谱 GLM-4），修改下面两行即可
llm = OpenAI(
    temperature=0.2,  # 越低越严谨
    model="gpt-4o-mini",
    openai_api_key="your-api-key-here",  # 换成你的 Key
    # openai_api_base="https://open.bigmodel.cn/api/paas/v4/" # 如果用智谱，取消注释
)

# ================= 2. 定义 5 个核心 Agent =================

# --- Agent 1: 文献检索与抓取 ---
def retrieve_agent(topic: str, max_num: int = 10):
    """从 ArXiv 抓取高相关文献的摘要和元数据"""
    print(f"[检索 Agent] 正在搜索主题: {topic} ...")
    
    # 构造 ArXiv 搜索查询 (限定 cs.CV 计算机视觉领域，近5年)
    search = arxiv.Search(
        query = f"ti:{topic} OR abs:{topic} AND cat:cs.CV",
        max_results = max_num,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": [str(a) for a in result.authors],
            "year": result.published.year,
            "abstract": result.summary,
            "url": result.entry_id
        })
    
    print(f"[检索 Agent] 成功获取 {len(papers)} 篇文献摘要。")
    return papers

# --- Agent 2: 文献精读与信息提取 ---
def create_extraction_chain():
    prompt = PromptTemplate(
        input_variables=["paper_text"],
        template="""
        你是一位专业的学术精读助手。请仔细阅读以下文献摘要，提取以下关键信息：
        
        文献摘要：
        {paper_text}
        
        请严格按 JSON 格式输出（不要输出其他文字）：
        {{
            "research_question": "本文研究的核心科学问题是什么？",
            "method": "本文采用的核心方法/技术路线是什么？",
            "conclusion": "本文的主要结论是什么？",
            "innovation": "本文的主要创新点是什么？"
        }}
        """
    )
    return LLMChain(llm=llm, prompt=prompt, output_key="extracted_info")

# --- Agent 3: 技术脉络推理 ---
def create_reasoning_chain():
    prompt = PromptTemplate(
        input_variables=["all_extracted"],
        template="""
        你是一位领域资深专家。以下是多篇文献的核心信息汇总：
        
        {all_extracted}
        
        请完成以下任务：
        1. 按时间顺序梳理该领域的技术演进路线（谁在什么时候提出了什么关键方法）。
        2. 指出该领域存在哪些主要的技术流派或研究分支。
        
        请用清晰的段落式中文回答。
        """
    )
    return LLMChain(llm=llm, prompt=prompt, output_key="tech_route")

# --- Agent 4: 研究缺口分析 ---
def create_gap_chain():
    prompt = PromptTemplate(
        input_variables=["tech_route", "all_extracted"],
        template="""
        基于以下技术脉络和文献信息：
        
        技术脉络：{tech_route}
        文献汇总：{all_extracted}
        
        请分析：
        1. 现有研究普遍存在哪些局限性？
        2. 还有哪些未被解决的关键科学问题或研究空白？
        
        请分点列出。
        """
    )
    return LLMChain(llm=llm, prompt=prompt, output_key="research_gaps")

# --- Agent 5: 格式校验与参考文献生成 ---
def create_format_chain():
    prompt = PromptTemplate(
        input_variables=["final_draft", "paper_list"],
        template="""
        你是一位专业的学术格式编辑。请执行以下操作：
        
        1. 为以下综述草稿添加一个简短的摘要（200字以内）。
        2. 在文末根据提供的文献列表，生成标准的 GB/T 7714-2015 格式参考文献。
        
        综述草稿：
        {final_draft}
        
        文献列表：
        {paper_list}
        
        请输出完整的、格式规范的综述全文。
        """
    )
    return LLMChain(llm=llm, prompt=prompt, output_key="formatted_review")

# ================= 3. 主流程：串联所有 Agent =================
def run_literature_review_system(topic: str = "YOLO object detection"):
    # 1. 执行检索
    raw_papers = retrieve_agent(topic, max_num=8)
    
    # 2. 执行精读 (遍历每一篇)
    extraction_chain = create_extraction_chain()
    all_extracted_data = []
    paper_basic_info_list = []
    
    print("[精读 Agent] 正在分析文献内容...")
    for idx, paper in enumerate(raw_papers):
        print(f"  正在分析第 {idx+1}/{len(raw_papers)} 篇: {paper['title'][:30]}...")
        result = extraction_chain.run(paper_text=paper["abstract"])
        all_extracted_data.append(f"文献 {idx+1}: {result}")
        
        # 保存基本信息用于最后生成参考文献
        paper_basic_info_list.append(f"[{idx+1}] {paper['title']}. {paper['authors'][0]}等. {paper['year']}. {paper['url']}")
    
    all_extracted_str = "\n".join(all_extracted_data)
    paper_list_str = "\n".join(paper_basic_info_list)
    
    # 3. 执行脉络推理
    print("[脉络推理 Agent] 正在梳理技术演进...")
    reasoning_chain = create_reasoning_chain()
    tech_route = reasoning_chain.run(all_extracted=all_extracted_str)
    
    # 4. 执行缺口分析
    print("[缺口分析 Agent] 正在识别研究空白...")
    gap_chain = create_gap_chain()
    gaps = gap_chain.run(tech_route=tech_route, all_extracted=all_extracted_str)
    
    # 5. 合并初稿
    draft = f"""
    # 文献综述：{topic}
    
    ## 1. 技术演进脉络
    {tech_route}
    
    ## 2. 研究局限与缺口
    {gaps}
    """
    
    # 6. 执行格式校验
    print("[格式 Agent] 正在生成最终文档...")
    format_chain = create_format_chain()
    final_review = format_chain.run(final_draft=draft, paper_list=paper_list_str)
    
    # 保存结果
    with open("literature_review_output.txt", "w", encoding="utf-8") as f:
        f.write(final_review)
    
    print("\n" + "="*50)
    print("✅ 系统运行完成！结果已保存至 literature_review_output.txt")
    print("="*50)
    print("\n最终预览：\n")
    print(final_review[:500] + "...")

# ================= 4. 运行系统 =================
if __name__ == "__main__":
    # 把这里改成你想研究的主题，比如 "Federated Learning" 或 "Segment Anything"
    YOUR_RESEARCH_TOPIC = "YOLO object detection"
    run_literature_review_system(YOUR_RESEARCH_TOPIC)