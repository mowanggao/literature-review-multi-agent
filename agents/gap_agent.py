from typing import List, Dict
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import Config
from tenacity import retry, stop_after_attempt, wait_exponential

class GapAgent:
    def __init__(self):
        self.llm = OpenAI(
            temperature=Config.TEMPERATURE,
            model=Config.OPENAI_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE
        )
        
        prompt_template = """
        你是一位具有前瞻性的学术研究人员。基于以下技术脉络和文献信息，请分析研究空白。
        
        技术脉络：
        {tech_evolution}
        
        文献详细信息：
        {extracted_papers}
        
        请完成以下分析：
        
        1. 现有研究局限性：系统总结当前研究普遍存在哪些不足之处？（分点列出，至少3点）
        
        2. 未解决问题：还有哪些关键科学问题或技术挑战尚未被充分研究？（分点列出，至少3点）
        
        3. 未来研究方向：基于以上分析，提出3-5个有前景的未来研究方向，并简要说明理由。
        
        请用清晰的中文回答，每点要有具体的论述。
        """
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["tech_evolution", "extracted_papers"]
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _format_extracted(self, extracted_papers: List[Dict]) -> str:
        formatted = []
        for idx, paper in enumerate(extracted_papers):
            info = paper["paper_info"]
            ext = paper["extracted"]
            formatted.append(f"""
            文献 {idx+1} ({info['year']}):
            - 局限性: {ext['limitations']}
            - 结论: {ext['conclusion']}
            """)
        return "\n".join(formatted)
    
    @retry(stop=stop_after_attempt(Config.MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, tech_evolution: str, extracted_papers: List[Dict]) -> Dict:
        print("[缺口分析 Agent] 正在识别研究空白...")
        
        formatted_papers = self._format_extracted(extracted_papers)
        result = self.chain.run(
            tech_evolution=tech_evolution,
            extracted_papers=formatted_papers
        )
        
        return {
            "research_gaps": result
        }
