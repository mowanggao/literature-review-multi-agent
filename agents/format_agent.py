from typing import List, Dict
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import Config
from datetime import datetime
import os

class FormatAgent:
    def __init__(self):
        self.llm = OpenAI(
            temperature=Config.TEMPERATURE,
            model=Config.OPENAI_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE
        )
        
        prompt_template = """
        你是一位专业的学术格式编辑。请对以下文献综述进行格式优化。
        
        综述主题：{topic}
        综述内容：
        {review_content}
        
        参考文献列表：
        {references}
        
        请执行以下操作：
        
        1. 添加中英文摘要：
           - 中文摘要：200-300字，概括全文内容
           - 英文摘要：200-300词，对应中文摘要
        
        2. 优化结构：
           - 确保各级标题层次清晰
           - 添加适当的过渡段落
        
        3. 生成参考文献：
           - 使用 GB/T 7714-2015 顺序编码制格式
           - 确保所有文献信息完整
        
        4. 添加引言和总结：
           - 引言：介绍研究背景和综述目的
           - 总结：概括全文主要发现
        
        请输出完整的、格式规范的综述全文，使用 Markdown 格式。
        """
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["topic", "review_content", "references"]
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _format_references_gbt7714(self, papers: List[Dict]) -> List[str]:
        references = []
        for idx, paper in enumerate(papers):
            authors = paper["authors"]
            author_str = ""
            if len(authors) >= 3:
                author_str = f"{authors[0]}, {authors[1]}, {authors[2]}等"
            elif len(authors) == 2:
                author_str = f"{authors[0]}, {authors[1]}"
            elif len(authors) == 1:
                author_str = authors[0]
            
            year = paper["year"]
            title = paper["title"]
            venue = paper.get("venue", "预印本") if paper.get("venue") else "预印本"
            url = paper["url"]
            
            ref = f"[{idx+1}] {author_str}. {title}[J]. {venue}, {year}. {url}"
            references.append(ref)
            
        return references
    
    def format(self, topic: str, tech_evolution: str, research_gaps: str, papers: List[Dict]) -> Dict:
        print("[格式 Agent] 正在生成最终文档...")
        
        review_content = f"""
        # 1. 引言
        
        本综述针对"{topic}"领域展开系统梳理...
        
        # 2. 技术演进脉络
        
        {tech_evolution}
        
        # 3. 研究局限与缺口
        
        {research_gaps}
        
        # 4. 总结
        
        综上所述...
        """
        
        references = self._format_references_gbt7714(papers)
        references_str = "\n".join(references)
        
        result = self.chain.run(
            topic=topic,
            review_content=review_content,
            references=references_str
        )
        
        return {
            "formatted_review": result,
            "references": references
        }
    
    def save_output(self, topic: str, content: str, output_dir: str = Config.OUTPUT_DIR):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic.replace(" ", "_").replace("/", "_")
        filename = f"literature_review_{safe_topic}_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"[格式 Agent] 结果已保存至: {filepath}")
        return filepath
