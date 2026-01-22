import asyncio
import time
import os

from src.advanced_rag.processor import AdvancedRAGProcessor

async def run_test():
    proc = AdvancedRAGProcessor()
    ok = proc.initialize()
    if not ok:
        print("Initialization failed")
        return

    query = "How can I establish a bedtime routine for a 2-year-old?"

    # Warmup: ensure cache is empty
    print("Running first retrieval (should populate cache)")
    t1 = time.time()
    results = await proc.multi_retrieval_async([query], None, retrieval_k=3)
    t2 = time.time()
    print(f"First retrieval time: {t2 - t1:.2f}s, docs sets: {[len(r) for r in results]}")

    print("Running second retrieval (should use cache)")
    t3 = time.time()
    results2 = await proc.multi_retrieval_async([query], None, retrieval_k=3)
    t4 = time.time()
    print(f"Second retrieval time: {t4 - t3:.2f}s, docs sets: {[len(r) for r in results2]}")

if __name__ == '__main__':
    asyncio.run(run_test())
