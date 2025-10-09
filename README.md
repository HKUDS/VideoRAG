<div align="center">
  <picture>
      <img src="cover.png" width="80%" style="border: none; box-shadow: none;" alt="Whizzhive: Chat with Your Videos">
  </picture>
  
  <h1>
    <strong>VideoRAG: Chat with Your Videos</strong> • <strong>Whizzhive Desktop</strong>
  </h1>
  
  <a href='https://arxiv.org/abs/2502.01549'><img src='https://img.shields.io/badge/arXiv-2502.01549-b31b1b'></a>
  <a href='https://github.com/HKUDS/VideoRAG/issues/1'><img src='https://img.shields.io/badge/群聊-wechat/feishu-green'></a>
  <a href='https://discord.gg/ZzU55kz3'><img src='https://discordapp.com/api/guilds/1296348098003734629/widget.png?style=shield'></a>
  <a href='https://www.youtube.com/watch?v=D5vsxcp4QZI'><img src='https://img.shields.io/badge/YouTube-Watch%20Demo-red?style=flat&logo=youtube'></a>
  [![Platform](https://img.shields.io/badge/platform-macOS%20|%20Windows%20|%20Linux-lightgrey.svg)]()

  **🎬 Intelligent Video Conversations | Powered by Advanced AI | Extreme Long-Context Processing**

</div>

<br/>

<img src='VideoRAG-algorithm/VideoRAG_cover.png' />

Whizzhive is a revolutionary desktop application that lets you **chat with your videos** using cutting-edge AI technology. Built on the powerful [VideoRAG framework](https://arxiv.org/abs/2502.01549), Whizzhive can understand and analyze videos of any length - from short clips to hundreds of hours of content - and answer your questions with remarkable accuracy.

### 🎥 Watch Whizzhive in Action

See how Whizzhive transforms video interaction with intelligent conversations and deep understanding capabilities.

<div align="center">
  <a href="https://www.youtube.com/watch?v=D5vsxcp4QZI">
    <img src="https://img.youtube.com/vi/D5vsxcp4QZI/maxresdefault.jpg" width="80%" alt="Whizzhive Introduction Video">
  </a>
  <p><em>👆 Click to watch the Whizzhive demo video</em></p>
</div>

## ✨ Key Features

### For Everyone
- **Drag & Drop Upload**: Simply drag video files into Whizzhive
- **Smart Conversations**: Ask questions in natural language
- **Multi-Format Support**: Works with MP4, MKV, AVI, and more
- **Cross-Platform**: Available on macOS, Windows, and Linux

### For Power Users
- **Extreme Long Videos**: Process videos up to hundreds of hours
- **Multi-Video Analysis**: Compare and analyze multiple videos simultaneously
- **Advanced Retrieval**: Find specific moments and scenes with precision
- **Export Capabilities**: Save insights and references for later use

### For Researchers
- **VideoRAG Framework**: Access to cutting-edge retrieval-augmented generation
- **Benchmark Dataset**: LongerVideos benchmark with 134+ hours of content
- **Performance Metrics**: Detailed evaluation against existing methods
- **Extensible Architecture**: Build upon our open-source foundation
  
## 🌟 Why Whizzhive?

**For Video Enthusiasts & Professionals:**
- **Effortless Video Analysis**: Upload any video and start asking questions immediately
- **Natural Conversations**: Chat with your videos as if talking to a human expert
- **No Length Limits**: Process everything from 30-second clips to 100+ hour documentaries
- **Deep Understanding**: Combines visual content, audio, and context for comprehensive answers

**For Researchers & Developers:**
- **State-of-the-Art Algorithm**: Built on VideoRAG, featuring graph-driven knowledge indexing
- **Benchmark Performance**: Evaluated on 134+ hours across lectures, documentaries, and entertainment
- **Open Source**: Full access to VideoRAG implementation and research findings
- **Scalable Architecture**: Efficient processing with single GPU (RTX 3090) capability

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [✨ Key Features](#-key-features)
- [🔬 VideoRAG Algorithm](#-videorag-algorithm)
- [🛠️ Development Setup](#️-development-setup)
- [🧪 Benchmarks & Evaluation](#-benchmarks--evaluation)
- [📖 Citation](#-citation)
- [🤝 Contributing](#-contributing)
- [🙏 Acknowledgement](#-acknowledgement)

## 🚀 Quick Start of Whizzhive

### Option 1: Download Whizzhive App (Coming Soon)

> [!NOTE]
> We are preparing the **Beta release** for macOS Apple Silicon first, with Windows and Linux versions coming soon!

<div align="left">
  <a href="https://github.com/HKUDS/Whizzhive/releases">
    <img src="https://img.shields.io/badge/Coming%20Soon-Mac%20Download-007ACC?style=for-the-badge&logo=apple&logoColor=white" alt="Coming Soon - Mac Release" height="50">
  </a>
</div>

### Option 2: Run from Source Code

For detailed setup instructions:

- **Whizzhive Desktop App**: See [Whizzhive-desktop](Whizzhive-desktop) for complete installation and configuration steps

**Quick Overview:**
1. Set up the Python backend environment and start the VideoRAG server
2. Launch the Electron frontend application
3. Start chatting with your videos!

## 🔬 VideoRAG Algorithm

<p align="center">
<img src="VideoRAG-algorithm/VideoRAG.png" alt="VideoRAG Architecture" width="80%" />
</p>

VideoRAG introduces a novel dual-channel architecture that combines:

- **Graph-Driven Knowledge Indexing**: Multi-modal knowledge graphs for structured video understanding
- **Hierarchical Context Encoding**: Preserves spatiotemporal visual patterns across long sequences  
- **Adaptive Retrieval**: Dynamic retrieval mechanisms optimized for video content
- **Cross-Video Understanding**: Semantic relationship modeling across multiple videos

### Technical Highlights

- **Efficient Processing**: Handle hundreds of hours on a single RTX 3090 (24GB)
- **Structured Indexing**: Distill long videos into concise knowledge representations
- **Multi-Modal Retrieval**: Align textual queries with visual and audio content
- **LongerVideos Benchmark**: 160+ videos, 134+ hours across diverse domains

### Performance Comparison

Our VideoRAG algorithm significantly outperforms existing methods in long-context video understanding:

<div align="center">
  <img src="Whizzhive-desktop/figures/table.png" width="80%" alt="Performance Comparison" />
</div>

### Experiments and Evaluation

See [VideoRAG-algorithm](VideoRAG-algorithm) for detailed development setup including:
- Conda environment creation
- Model checkpoints download
- Dependencies installation
- Evaluation scripts

## 🧪 LongerVideos Benchmark

We created the LongerVideos benchmark to evaluate long-context video understanding:

| Video Type       | #Collections | #Videos | #Queries | Avg. Duration |
|------------------|-------------|---------|----------|---------------|
| **Lectures**     | 12          | 135     | 376      | ~64.3 hours   |
| **Documentaries**| 5           | 12      | 114      | ~28.5 hours   |
| **Entertainment**| 5           | 17      | 112      | ~41.9 hours   |
| **Total**        | 22          | 164     | 602      | ~134.6 hours  |

For detailed evaluation instructions and reproduction scripts, see [VideoRAG-algorithm/reproduce](VideoRAG-algorithm/reproduce).

## 📖 Citation

If you find Whizzhive or VideoRAG helpful in your research, please cite our paper:

```bibtex
@article{VideoRAG,
  title={VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos},
  author={Ren, Xubin and Xu, Lingrui and Xia, Long and Wang, Shuaiqiang and Yin, Dawei and Huang, Chao},
  journal={arXiv preprint arXiv:2502.01549},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions from the community! Whether you're:

- **Reporting bugs** or suggesting features for Whizzhive
- **Improving VideoRAG algorithms** or adding new capabilities  
- **Enhancing documentation** or creating tutorials
- **Designing UI/UX improvements** for better user experience

Feel free to submit issues and pull requests. Together, we're building the future of intelligent video interaction!

## 🙏 Acknowledgement

Whizzhive builds upon the incredible work of the open-source community:

- **[VideoRAG](https://arxiv.org/abs/2502.01549)**: The core algorithm powering Whizzhive's intelligence
- **[nano-graphrag](https://github.com/gusye1234/nano-graphrag)** & **[LightRAG](https://github.com/HKUDS/LightRAG)**: Graph-based retrieval foundations
- **[ImageBind](https://github.com/facebookresearch/ImageBind)**: Multi-modal representation learning
- **[uitars-desktop](https://github.com/bytedance/UI-TARS-desktop)**: Desktop application architecture inspiration

**🌟 Transform how you interact with videos. Start your journey with Whizzhive today!**

---

<div align="center">
  <sub>Built with ❤️ by the VideoRAG@HKUDS team.</sub>
</div> 
