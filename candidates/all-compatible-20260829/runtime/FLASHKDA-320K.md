# FlashKDA 320K production trial

- Image: `sha256:087276f1ec791909e71055d655eeadb3b35c246e46f2be21af9d210aa970a60b`
- KDA backend: FlashKDA
- Maximum context: 320,000 tokens
- Fixed GPU KV reservation: 1,879,048,192 bytes (1.75 GiB/rank)
- Scheduler target/draft budgets: 3,080 / 3,080
- Rollback: `compose.tp8-dflash-compatible-sm120-triton.yml`

The 52,000-token context reduction keeps approximately two maximum-length
requests within the expected checkpoint-aware KV capacity while the 256 MiB
KV reduction creates real working memory for FlashKDA. The draft server keeps
its existing 372,736-token retained window; a larger draft limit is safe when
the target rejects prompts above 320,000 tokens.

This is the active production profile. It exposes 625,777 KV tokens (1.96x a
maximum-length request), uses the SM120 TensorMap FlashKDA binary with SHA-256
`c0b440ab87e6cfec16b23138f95a67d8fe9142836c040a56c41a574dc7d27ed1`,
and passed normal, structured, four-way concurrent, disconnect/retry, 16K
checkpoint replay, and 32K cold-prefill gates without CUDA errors.
