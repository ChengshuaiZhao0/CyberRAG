# CyberBot

This is GitHub Repo for paper "[Ontology-Aware RAG for Improved Question-Answering in Cybersecurity Education](https://arxiv.org/abs/2412.14191)"

![image-20241222221611937](fig/overview.jpg)

## Structure

The project overview is listed as following:

- **dataset**: includes knowledge base used for RAG, KG ontology for the answer validation, and QA dataset for experiment.
- **answer_retriever.py**: includes the retriever of the RAG.
- **llm.py**: includes the LLM.
- **main.py**: includes the experiments.
- **test.py**: includes some intermediate processing procedure
- **utils.py**: includes the utils.

## Case Study

![case_study](fig/case_study.jpg)

## Reference

```tex
@article{zhao2024ontology,
  title={Ontology-Aware RAG for Improved Question-Answering in Cybersecurity Education},
  author={Zhao, Chengshuai and Agrawal, Garima and Kumarage, Tharindu and Tan, Zhen and Deng, Yuli and Chen, Ying-Chih and Liu, Huan},
  journal={arXiv preprint arXiv:2412.14191},
  year={2024}
}
```
