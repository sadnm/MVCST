# MVCST: Integrating Cross-Slice Spatial Transcriptomics via Multi-View Contrastive Representation Learning

## Requirements

Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies listed in `requirements.txt` include:

- `numpy==1.24.3`
- `scanpy==1.9.3`
- `anndata==0.9.1`
- `pandas==2.0.1`
- `scipy==1.10.1`
- `scikit-learn==1.2.2`
- `tqdm==4.65.0`
- `matplotlib==3.7.1`
- `seaborn==0.12.2`
- `pot==0.9.5`
- `scib==1.1.7`
- `torch_geometric`

## Original Data Sources

All datasets used in this study are publicly available.

| Dataset | Original Data Source |
| --- | --- |
| Human DLPFC dataset | [spatialLIBD](http://spatial.libd.org/spatialLIBD) |
| Mouse brain dataset | [10x Genomics Spatial Gene Expression Datasets](https://support.10xgenomics.com/spatial-gene-expression/datasets) |
| Mouse olfactory bulb datasets | [SEDR analyses](https://github.com/JinmiaoChenLab/SEDR_analyses) and [Broad Institute SCP815](https://singlecell.broadinstitute.org/single_cell/study/SCP815) |
| Mouse hippocampus datasets | [Broad Institute SCP815](https://singlecell.broadinstitute.org/single_cell/study/SCP815) and [Broad Institute SCP1663](https://singlecell.broadinstitute.org/single_cell/study/SCP1663) |
| Human developmental heart dataset | [Developmental_heart](https://github.com/MickanAsp/Developmental_heart) |

## Download Data

### Download from Hugging Face (Recommended)

We have hosted preprocessed datasets on Hugging Face for easier access:

Repository: [lanyu1/mvcst_data](https://huggingface.co/datasets/lanyu1/mvcst_data)

Compressed archive: [`dataset.zip`](https://huggingface.co/datasets/lanyu1/mvcst_data/resolve/main/dataset.zip)
