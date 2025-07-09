#!/usr/bin/env python3
"""
RAG 시스템 디버깅 도구
"""

import sys
import os
sys.path.append('.')

from rag.retriever import FAISSRetriever
from rag.embedder import BGEEmbedder
import yaml

def debug_rag():
    """RAG 시스템 디버깅"""
    config_path = "llm/config.yaml"
    index_path = "models/faiss_index"
    
    print("=== RAG 시스템 디버깅 ===")
    
    # 1. 설정 확인
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"1. 설정 확인:")
    print(f"   - similarity_threshold: {config['rag']['similarity_threshold']}")
    print(f"   - top_k: {config['rag']['top_k']}")
    print(f"   - chunk_size: {config['rag']['chunk_size']}")
    
    # 2. 임베딩 모델 로드 테스트
    print("\n2. 임베딩 모델 로드 테스트:")
    try:
        embedder = BGEEmbedder(config_path)
        print(f"   ✓ 임베딩 모델 로드 성공")
        print(f"   - 모델명: {config['embedding']['model_name']}")
        print(f"   - 임베딩 차원: {embedder.get_embedding_dimension()}")
    except Exception as e:
        print(f"   ✗ 임베딩 모델 로드 실패: {e}")
        return
    
    # 3. 인덱스 로드 테스트
    print("\n3. 인덱스 로드 테스트:")
    try:
        retriever = FAISSRetriever(config_path)
        retriever.load_index(index_path)
        print(f"   ✓ 인덱스 로드 성공")
        stats = retriever.get_stats()
        for key, value in stats.items():
            print(f"   - {key}: {value}")
    except Exception as e:
        print(f"   ✗ 인덱스 로드 실패: {e}")
        return
    
    # 4. 검색 테스트
    print("\n4. 검색 테스트:")
    test_queries = [
        "AlphaGo defeats Go world Champion",
        "When did AlphaGo defeat Go champion",
        "artificial intelligence",
        "machine learning types",
        "2016 AlphaGo"
    ]
    
    for query in test_queries:
        print(f"\n   쿼리: '{query}'")
        try:
            # 원본 threshold 저장
            original_threshold = config['rag']['similarity_threshold']
            
            # 매우 낮은 threshold로 테스트
            config['rag']['similarity_threshold'] = 0.1
            
            # 검색 실행
            results = retriever.search(query, top_k=3)
            
            if results:
                print(f"   ✓ {len(results)}개 결과 발견:")
                for i, (doc_text, similarity, metadata) in enumerate(results, 1):
                    filename = metadata.get('filename', 'Unknown')
                    print(f"      {i}. {filename} (유사도: {similarity:.4f})")
                    print(f"         텍스트: {doc_text[:100]}...")
            else:
                print(f"   ✗ 검색 결과 없음")
            
            # threshold 복원
            config['rag']['similarity_threshold'] = original_threshold
            
        except Exception as e:
            print(f"   ✗ 검색 실패: {e}")
    
    # 5. 문서 내용 확인
    print("\n5. 인덱싱된 문서 내용 샘플:")
    try:
        if retriever.documents:
            for i, doc in enumerate(retriever.documents[:3]):  # 처음 3개 청크만 표시
                metadata = retriever.document_metadata[i]
                filename = metadata.get('filename', 'Unknown')
                print(f"   문서 {i+1} ({filename}):")
                print(f"      {doc[:150]}...")
        else:
            print("   ✗ 인덱싱된 문서 없음")
    except Exception as e:
        print(f"   ✗ 문서 내용 확인 실패: {e}")
    
    print("\n=== 디버깅 완료 ===")

if __name__ == "__main__":
    debug_rag() 