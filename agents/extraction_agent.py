from typing import Dict, List
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config import Config
from tenacity import retry, stop_after_attempt, wait_exponential

class PaperExtraction(BaseModel):
    research_question: str = Field(description="本文研究的核心科学问题")
    method: str = Field(description="本文采用的核心方法/技术路线")
    conclusion: str = Field(description="本文的主要结论")
    innovation: str = Field(description="本文的主要创新点")
    limitations: str = Field(description="本文提到的局限性（如有）")

class ExtractionAgent:
    def __init__(self):
        self.llm = OpenAI(
            temperature=Config.TEMPERATURE,
            model=Config.OPENAI_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE
        )
        self.parser = PydanticOutputParser(pydantic_object=PaperExtraction)
        
        prompt_template = """
        你是一位专业的学术精读助手。请仔细阅读以下文献摘要，提取关键信息。
        
        文献标题：{title}
        文献摘要：{abstract}
        
        {format_instructions}
        """
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["title", "abstract"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    @retry(stop=stop_after_attempt(Config.MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract(self, paper: Dict) -> Dict:
        try:
            result = self.chain.run(
                title=paper["title"],
                abstract=paper["abstract"]
            )
            parsed = self.parser.parse(result)
            
            return {
                "paper_info": paper,
                "extracted": {
                    "research_question": parsed.research_question,
                    "method": parsed.method,
                    "conclusion": parsed.conclusion,
                    "innovation": parsed.innovation,
                    "limitations": parsed.limitations
                }
            }
        except Exception as e:
            print(f"[精读 Agent] 解析失败，返回默认结构: {str(e)}")
            return {
                "paper_info": paper,
                "extracted": {
                    "research_question": "无法提取",
                    "method": "无法提取",
                    "conclusion": "无法提取",
                    "innovation": "无法提取",
                    "limitations": "无"
                }
            }
    
    def extract_batch(self, papers: List[Dict]) -> List[Dict]:
        print(f"[精读 Agent] 开始分析 {len(papers)} 篇文献...")
        extracted_papers = []
        
        for idx, paper in enumerate(papers):
            print(f"  正在分析第 {idx+1}/{len(papers)} 篇: {paper['title'][:40]}...")
            extracted = self.extract(paper)
            extracted_papers.append(extracted)
            
        print(f"[精读 Agent] 完成 {len(extracted_papers)} 篇文献分析")
        return extracted_papers
