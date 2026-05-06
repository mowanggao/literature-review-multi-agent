import os
import sys
import json
from datetime import datetime
from agents import (
    RetrievalAgent,
    ExtractionAgent,
    ReasoningAgent,
    GapAgent,
    FormatAgent
)
from config import Config

class LiteratureReviewSystem:
    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.extraction_agent = ExtractionAgent()
        self.reasoning_agent = ReasoningAgent()
        self.gap_agent = GapAgent()
        self.format_agent = FormatAgent()
        
    def run(self, topic: str, max_papers: int = 15, years: int = 5):
        print("=" * 60)
        print("  学术文献综述多 Agent 协作系统")
        print("=" * 60)
        print(f"  主题: {topic}")
        print(f"  文献数量: {max_papers}")
        print(f"  时间范围: 最近 {years} 年")
        print("=" * 60)
        print()
        
        start_time = datetime.now()
        
        try:
            papers = self.retrieval_agent.retrieve(topic, max_papers, years)
            if not papers:
                print("错误: 未获取到任何文献")
                return None
            print()
            
            extracted_papers = self.extraction_agent.extract_batch(papers)
            print()
            
            reasoning_result = self.reasoning_agent.reason(extracted_papers)
            tech_evolution = reasoning_result["tech_evolution"]
            print()
            
            gap_result = self.gap_agent.analyze(tech_evolution, extracted_papers)
            research_gaps = gap_result["research_gaps"]
            print()
            
            format_result = self.format_agent.format(
                topic, 
                tech_evolution, 
                research_gaps, 
                papers
            )
            formatted_review = format_result["formatted_review"]
            
            output_path = self.format_agent.save_output(topic, formatted_review)
            self._save_intermediate(topic, papers, extracted_papers, tech_evolution, research_gaps)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print()
            print("=" * 60)
            print("  系统运行完成！")
            print(f"  总耗时: {duration:.2f} 秒")
            print(f"  输出文件: {output_path}")
            print("=" * 60)
            
            return {
                "topic": topic,
                "papers": papers,
                "extracted_papers": extracted_papers,
                "tech_evolution": tech_evolution,
                "research_gaps": research_gaps,
                "formatted_review": formatted_review,
                "output_path": output_path
            }
            
        except Exception as e:
            print(f"错误: 系统运行失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_intermediate(self, topic, papers, extracted_papers, tech_evolution, research_gaps):
        if not os.path.exists(Config.OUTPUT_DIR):
            os.makedirs(Config.OUTPUT_DIR)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic.replace(" ", "_").replace("/", "_")
        
        data = {
            "topic": topic,
            "timestamp": timestamp,
            "papers": papers,
            "extracted_papers": extracted_papers,
            "tech_evolution": tech_evolution,
            "research_gaps": research_gaps
        }
        
        filepath = os.path.join(Config.OUTPUT_DIR, f"intermediate_{safe_topic}_{timestamp}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"[系统] 中间结果已保存至: {filepath}")

def main():
    topic = "YOLO object detection"
    
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    
    system = LiteratureReviewSystem()
    result = system.run(topic, max_papers=12, years=5)
    
    if result:
        print()
        print("预览:")
        print("-" * 60)
        preview = result["formatted_review"][:800]
        print(preview)
        print("...")
        print("-" * 60)

if __name__ == "__main__":
    main()
