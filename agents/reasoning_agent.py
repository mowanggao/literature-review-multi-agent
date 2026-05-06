from typing import List, Dict
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import Config
from tenacity import retry, stop_after_attempt, wait_exponential

class ReasoningAgent:
    def __init__(self):
        self.llm = OpenAI(
            temperature=Config.TEMPERATURE,
            model=Config.OPENAI_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE
        )
        
        prompt_template = """
        你是一位领域资深专家。以下是多篇文献的核心信息汇总：
        
        {extracted_papers}
        
        请完成以下任务：
        
        1. 技术演进路线：按时间顺序梳理该领域的技术发展脉络，说明谁在什么时候提出了什么关键方法，以及这些方法之间的继承关系。
        
        2. 研究流派分析：指出该领域存在哪些主要的技术流派或研究分支，各流派的核心观点和代表工作是什么。
        
        3. 关键里程碑：列出该领域发展过程中的3-5个最重要的里程碑式工作。
        
        请用清晰的段落式中文回答，结构分明。
        """
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["extracted_papers"]
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _format_extracted(self, extracted_papers: List[Dict]) -> str:
        formatted = []
        for idx, paper in enumerate(extracted_papers):
            info = paper["paper_info"]
            ext = paper["extracted"]
            formatted.append(f"""
            --- 文献 {idx+1} ---
            标题: {info['title']}
            作者: {', '.join(info['authors'][:3])}{'等' if len(info['authors']) > 3 else ''}
            年份: {info['year']}
            研究问题: {ext['research_question']}
            方法: {ext['method']}
            结论: {ext['conclusion']}
            创新点: {ext['innovation']}
            """)
        return "\n".join(formatted)
    
    @retry(stop=stop_after_attempt(Config.MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
    def reason(self, extracted_papers: List[Dict]) -> Dict:
        print("[脉络推理 Agent] 正在梳理技术演进路线...")
        
        formatted_input = self._format_extracted(extracted_papers)
        result = self.chain.run(extracted_papers=formatted_input)
        
        return {
            "tech_evolution": result
        }
