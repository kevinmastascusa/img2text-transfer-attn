# img2text-transfer-attn
img2text-transfer-attn is a modular image-captioning codebase for MS COCO. A pre-trained CNN encodes images into feature embeddings, which are decoded by either an LSTM or a Transformer with masked multi-head attention and positional encodings. Easily swap encoders or fine-tune layers.

[![Course: DSCI-471 SP24-25](https://img.shields.io/badge/Course-DSCI--471%20SP24--25-blueviolet.svg)](./)
[![Dataset: MS COCO](https://img.shields.io/badge/Dataset-MS%20COCO-blue.svg)](https://cocodataset.org/)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Project Status: WIP](https://img.shields.io/badge/Status-Work%20In%20Progress-orange.svg)](./)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)


## Architecture

The following diagram illustrates the architecture of the Image-to-Text Generation system, detailing both the baseline (CNN + LSTM) and advanced (CNN + Transformer) models.

```mermaid
graph LR
    A["Input Image (RGB Tensor) from MS COCO Dataset"] --> B{"Pre-trained CNN Encoder"};
    B -- "Image Features" --> C["Image Embedding"];

    subgraph "Baseline Model (CNN + LSTM)"
        C --> D["LSTM Decoder"];
        D -- "Generates" --> E("Caption Sequence - Baseline");
    end

    subgraph "Advanced Model (CNN + Transformer)"
        C --> F["Transformer Decoder"];
        F_PE["Positional Encoding"] --> F;
        F_MMA["Masked Multi-Head Self-Attention"] --> F;
        F -- "Generates" --> G("Caption Sequence - Advanced");
    end

    subgraph "Evaluation & Analysis"
        E --> H{"Evaluation Metrics (BLEU, METEOR, CIDEr)"};
        G --> H;
        H --> I["Performance Comparison & Findings"];
        E --> J["Qualitative Analysis (Example Captions)"];
        G --> J;
        F_MMA --> K["Attention Map Visualization"];
        J --> I;
        K --> I;
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style B fill:#b3e0ff,stroke:#333,stroke-width:2px,color:#000
    style C fill:#c1f0c1,stroke:#333,stroke-width:2px,color:#000
    style D fill:#ffd699,stroke:#333,stroke-width:2px,color:#000
    style E fill:#ffffb3,stroke:#333,stroke-width:2px,color:#000
    style F fill:#ffcc99,stroke:#333,stroke-width:2px,color:#000
    style F_PE fill:#ffe0b3,stroke:#333,stroke-width:1px,color:#000
    style F_MMA fill:#ffe0b3,stroke:#333,stroke-width:1px,color:#000
    style G fill:#ffffb3,stroke:#333,stroke-width:2px,color:#000
    style H fill:#ffb3b3,stroke:#333,stroke-width:2px,color:#000
    style I fill:#e0e0e0,stroke:#333,stroke-width:2px,color:#000
    style J fill:#d9ead3,stroke:#333,stroke-width:2px,color:#000
    style K fill:#d9ead3,stroke:#333,stroke-width:2px,color:#000
```
