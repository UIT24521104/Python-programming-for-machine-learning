## **Competition: SIIM-ISIC Melanoma Classification**
### Leaderboard source code:
### **Rank:** 1st Place
- **Authors:** Bo, Qishen Ha, Gary
- **GitHub Repository:** [https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution)

Technical Highlights
- **Architecture:** Utilized advanced networks such as EfficientNet (B0-B6) and ResNeSt.
- **Data:** Leveraged external datasets (ISIC 2018-2019) for pre-training.
- **Techniques:** - Heavy augmentation (Mixup, Cutout).
    - Optimized Loss function (BCE with Logits).
    - Implemented Test Time Augmentation (TTA).
### **Rank:** 2nd Place
- **Author:** Ian Pan
- **GitHub Repository:** [https://github.com/i-pan/kaggle-melanoma](https://github.com/i-pan/kaggle-melanoma)

Technical Highlights
- **Model:** Focused on EfficientNet architectures, specifically optimized for high-resolution images.
- **Strategy:** - Rigorous K-Fold Cross-Validation to ensure model reliability.
    - Handled class imbalance through custom sampling techniques.
    - Used model ensembling to boost Public/Private Leaderboard scores.