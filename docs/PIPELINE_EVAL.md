# Pipeline evaluation

Profile **`ci-retrieval-eval`** scores synthetic retrieval simulation:

- `success_rate` — golden checks pass for `recursive_hybrid_single`
- `mean_recall_at_k` — labeled relevant chunks appear in top-k
- `hybrid_recall_at_k` — hybrid-tagged cases recall
- `metadata_filter_accuracy` — metadata filters respected in results
- `tenant_isolation_rate` — tenant partition strategy keeps foreign tenants out
- `max_latency_ms_p95` — tail latency budget (simulated)

Run:

```bash
retrieval-lab evaluate-pipeline --profile ci-retrieval-eval --output reports/measure-ci-retrieval-eval.json
retrieval-lab check-measure --profile ci-retrieval-eval --report reports/measure-ci-retrieval-eval.json
```

Downstream **reranking** is covered in [reranking-strategies](https://github.com/CaioHAndradeLima/reranking-strategies); pass retrieved candidate IDs into that lab after this stage.