# Toxic Comment Sentiment Analysis Pipeline
The pipeline for toxic comment sentiment analysis

## Imbalanced dataset

### Train dataset from vihsd dataset 
|Clean  | Toxic |
| :---: | :---: |
| 18653| 4014  |

### Upsampling dataset by synonym replacement (SR)
Adding synonym words in /synonyms_upsampling/syn.json
Adding sample text in /synonyms_upsampling/text_sample_toxic.txt

Run command line 
```
python upsample_text.py
python concat_file.py
```
### Dataset distributions after upsampling data 
|Clean  | Toxic |
| :---: | :---: |
| 13370| 11714  |

## Fine-tune phoBERT for binary classification task toxic - non-toxic 
### Experiments
Training with lr=3e-5, batch_size=64, epoch=10, fold=5
Accuracy: 0.85
|Clean  | Toxic |
| :---: | :---: |
| 0.87 | 0.77 |

## Deploy model using streamlit
Command Line 
```
streamlit run deploy_streamlit.py
```

## Deploy model as api using fastapi 
Command Line 
```
uvicorn main_bitoxic:app --reload   
```
