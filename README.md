<div align="center">

# VideoRAG

**Retrieval-augmented generation for extreme long-context video understanding**

[![arXiv](https://img.shields.io/badge/arXiv-2502.01549-b31b1b)](https://arxiv.org/abs/2502.01549)
[![Discord](https://img.shields.io/discord/1296348098003734629?label=discord)](https://discord.gg/ZzU55kz3)

</div>

This repository contains the **VideoRAG** research implementation ([paper](https://arxiv.org/abs/2502.01549)), helper scripts at the repo root, and an optional **Vimo Desktop** app for chatting with videos.

## Documentation

| Doc | Purpose |
|-----|---------|
| **[VideoRAG_algorithm/README.md](VideoRAG_algorithm/README.md)** | **Main guide** — installation, dependencies, model checkpoints, quick start, LongerVideos, evaluation, citation |
| [Vimo-desktop](Vimo-desktop) | Desktop UI (Electron) and app-specific setup |

Algorithm details, conda environment, `pip` installs, and Python examples live under **`VideoRAG_algorithm/`**. Start there to run indexing and queries.

## Repository layout

```
VideoRAG/
├── VideoRAG_algorithm/     # Core library (videorag package), benchmarks, reproduce scripts
├── Vimo-desktop/           # Optional desktop client
├── construct_graph.py      # Example: index videos (Gemini config in this fork)
├── ask_graph.py            # Example: query an indexed graph
├── setup.sh                # Local environment helper (if present)
├── .checkpoints/           # ImageBind weights (see algorithm README)
├── MiniCPM-V-2_6-int4/     # Caption model (git LFS / Hugging Face)
└── faster-distil-whisper-large-v3/   # ASR model
```

Large model directories are listed in `.gitignore`; download checkpoints as described in [VideoRAG_algorithm/README.md](VideoRAG_algorithm/README.md).

## Quick pointer

1. Follow **[VideoRAG_algorithm/README.md](VideoRAG_algorithm/README.md)** for environment setup and checkpoints.
2. Configure API keys / LLM settings as required by your fork (e.g. Gemini or OpenAI in `VideoRAG_algorithm/videorag/_llm.py`).
3. Run examples from the algorithm package or use the root `construct_graph.py` / `ask_graph.py` with `PYTHONPATH` including this repository root if imports use the `VideoRAG_algorithm` package path.

## Citation

```bibtex
@article{VideoRAG,
  title={VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos},
  author={Ren, Xubin and Xu, Lingrui and Xia, Long and Wang, Shuaiqiang and Yin, Dawei and Huang, Chao},
  journal={arXiv preprint arXiv:2502.01549},
  year={2025}
}
```

## License

See [LICENSE](LICENSE).
