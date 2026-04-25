"""Judge prompts used by DatBench release tooling."""

VQA_V2_SEMANTIC_JUDGE_PROMPT = """You are judging a VQA answer for semantic correctness.

Given a question, accepted ground-truth answers, and a model response, decide whether the model response should be counted as correct.

Accept answers that preserve the same meaning as the ground truth, including harmless paraphrases, articles, pluralization, casing, punctuation, or longer phrases that clearly contain the answer.

Be strict about yes/no polarity, numeric/count answers, colors, object identity, spatial relations, OCR text, proper nouns, and any extra specificity that changes the answer.

If the model gives multiple conflicting answers, judge it incorrect unless the final answer is unambiguous and correct.

Return only JSON with this schema:
{"answer": true} or {"answer": false}"""
