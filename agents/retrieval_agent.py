import arxiv
import requests
from typing import List, Dict
from datetime import datetime, timedelta
import time

class RetrievalAgent:
    def __init__(self):
        pass
    
    def search_arxiv(self, topic: str, max_num: int = 10, years: int = 5) -> List[Dict]:
        print(f"[检索 Agent] 正在从 ArXiv 搜索主题: {topic}")
        papers = []
        try:
            cutoff_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d%H%M%S")
            search = arxiv.Search(
                query=f"ti:{topic} OR abs:{topic}",
                max_results=max_num,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for idx, result in enumerate(search.results()):
                papers.append({
                    "id": f"arxiv_{idx+1}",
                    "title": result.title,
                    "authors": [str(a) for a in result.authors],
                    "year": result.published.year,
                    "abstract": result.summary,
                    "url": result.entry_id,
                    "doi": result.doi if result.doi else "",
                    "source": "arXiv"
                })
                
            print(f"[检索 Agent] ArXiv 成功获取 {len(papers)} 篇文献")
        except Exception as e:
            print(f"[检索 Agent] ArXiv 搜索失败: {str(e)}")
            
        return papers
    
    def search_semantic_scholar(self, topic: str, max_num: int = 10, years: int = 5) -> List[Dict]:
        print(f"[检索 Agent] 正在从 Semantic Scholar 搜索主题: {topic}")
        papers = []
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": topic,
                "limit": max_num,
                "fields": "title,abstract,authors,year,url,venue,citationCount"
            }
            headers = {"Accept": "application/json"}
            
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for idx, paper in enumerate(data.get("data", [])):
                    if paper.get("year") and paper["year"] >= (datetime.now().year - years):
                        authors = [a.get("name", "") for a in paper.get("authors", [])]
                        papers.append({
                            "id": f"s2_{idx+1}",
                            "title": paper.get("title", ""),
                            "authors": authors,
                            "year": paper.get("year"),
                            "abstract": paper.get("abstract", ""),
                            "url": paper.get("url", ""),
                            "doi": paper.get("externalIds", {}).get("DOI", ""),
                            "venue": paper.get("venue", ""),
                            "citations": paper.get("citationCount", 0),
                            "source": "Semantic Scholar"
                        })
                        
            print(f"[检索 Agent] Semantic Scholar 成功获取 {len(papers)} 篇文献")
            time.sleep(1)
        except Exception as e:
            print(f"[检索 Agent] Semantic Scholar 搜索失败: {str(e)}")
            
        return papers
    
    def retrieve(self, topic: str, max_num: int = 20, years: int = 5) -> List[Dict]:
        arxiv_papers = self.search_arxiv(topic, max_num // 2, years)
        s2_papers = self.search_semantic_scholar(topic, max_num // 2, years)
        
        all_papers = arxiv_papers + s2_papers
        
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            title_lower = paper["title"].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
                
        unique_papers.sort(key=lambda x: x.get("citations", 0) if x.get("citations") else x["year"], reverse=True)
        
        print(f"[检索 Agent] 共获取 {len(unique_papers)} 篇去重文献")
        return unique_papers[:max_num]
