# FRAG - Framework for Retrieval Augmented Generation Evaluation and Benchmarking

Introducing FRAG - Framework for Retrieval Augmented Generation evaluating and benchmarking.

Current common LLM evaluation suites are based evaluated based on tasks that are used as proxy for intelligence and based on styles. For example, Grade School Math (GSM8k), Massive Multitask Language Understanding (MMLU, k=5). Or for styles, for examples, LMSys that uses LLM as a proxy for human evaluation. While they might proxy LLM's intelligence, they do not evaluate LLM's capability for production use cases.

For production basis, we would want to evaluate based on their capabilities of their production use cases, specifically, hallucinations, context utilization, instruction following, tool dependents potentials used by LLM applications with production usage in mind.

## Benchmarking LLM API Endpoints

We are using `httpx`.

## Evaluating Factuality Benchmarking online serving throughput for LLM API endpoints.



1. Context Utilization
- https://github.com/stunningpixels/lou-eval
- https://github.com/run-llama/llama_index/blob/main/docs/examples/response_synthesizers/long_context_test.ipynb
- https://arxiv.org/abs/2310.01427
- https://github.com/gkamradt/LLMTest_NeedleInAHaystack


2. Hallucination
- https://github.com/vectara/hallucination-leaderboard
- https://www.rungalileo.io/hallucinationindex
- https://arxiv.org/abs/2310.18344
- https://www.nature.com/articles/s41598-023-41032-5#MOESM1

3. Instruction following
- https://github.com/google-research/google-research/tree/master/instruction_following_eval

4. Table understanding
- https://osu-nlp-group.github.io/TableLlama/

4. Tool usage
-

# References

- [Reproducible Performance Metrics for LLM inference](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference)
